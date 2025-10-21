package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

const (
	shopifyAPIVersion     = "2024-07"
	defaultModel          = "gpt-5-nano-2025-08-07"
	defaultModels         = "gpt-5-nano-2025-08-07"
	defaultProductsLimit  = 50
	defaultThrottleMillis = 750
	defaultTemperature    = 0.2
	maxOutputTokens       = 180
	defaultOpenAIWorkers  = 4
	defaultMinScore       = 70
)

type config struct {
	ShopifyStore       string
	ShopifyAccessToken string
	OpenAIKey          string
	Model              string
	ModelsCSV          string
	ModelCompare       bool
	ExamplesPath       string
	MinScore           int
	ReportPath         string
	HandleFilter       string
	LimitImages        int
	ProductsLimit      int
	Temperature        float64
	Throttle           time.Duration
	OpenAIWorkers      int
}

type runner struct {
	cfg        config
	shopify    *shopifyClient
	openai     *openAIClient
	dryRun     bool
	force      bool
	httpClient *http.Client
}

func main() {
	var (
		dryRunFlag      = flag.Bool("dry", false, "preview changes without writing to Shopify")
		forceFlag       = flag.Bool("force", false, "overwrite existing alt text")
		throttleStr     = flag.String("throttle", fmt.Sprintf("%dms", defaultThrottleMillis), "per-request delay (e.g., 500ms, 1s)")
		modelsFlag      = flag.String("models", "", "comma-separated model fallbacks (first wins)")
		modelCompareFlg = flag.Bool("model-compare", false, "generate alt text with all models and print comparison")
		examplesPath    = flag.String("examples", "", "path to JSON/CSV with few-shot examples to inject")
		minScore        = flag.Int("min-score", defaultMinScore, "minimum quality score to write to Shopify (0-100)")
		reportPath      = flag.String("report", "", "write JSONL report of generations (inputs, outputs, scores)")
		handleFilter    = flag.String("handles", "", "comma-separated product handle substrings to filter")
		limitImages     = flag.Int("limit-images", 0, "max images to process (for testing)")
	)
	flag.Parse()

	cfg, err := loadConfig(*throttleStr)
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	// Apply flag overrides
	if *modelsFlag != "" {
		cfg.ModelsCSV = *modelsFlag
	}
	cfg.ModelCompare = *modelCompareFlg
	cfg.ExamplesPath = *examplesPath
	cfg.MinScore = *minScore
	cfg.ReportPath = *reportPath
	cfg.HandleFilter = *handleFilter
	cfg.LimitImages = *limitImages

	httpClient := &http.Client{Timeout: 60 * time.Second}
	shopify := newShopifyClient(cfg.ShopifyStore, cfg.ShopifyAccessToken, httpClient)

	// Load few-shot examples if provided
	fewShots, err := loadFewShot(cfg.ExamplesPath)
	if err != nil {
		log.Fatalf("examples error: %v", err)
	}
	if len(fewShots) > 0 {
		log.Printf("Loaded %d few-shot examples from %s", len(fewShots), cfg.ExamplesPath)
	}

	openai := newOpenAIClient(cfg.OpenAIKey, cfg.Model, splitCSV(cfg.ModelsCSV), cfg.ModelCompare, cfg.Temperature, fewShots, httpClient)

	r := &runner{
		cfg:        cfg,
		shopify:    shopify,
		openai:     openai,
		dryRun:     *dryRunFlag,
		force:      *forceFlag,
		httpClient: httpClient,
	}

	ctx := context.Background()
	if err := r.run(ctx); err != nil {
		log.Fatalf("run error: %v", err)
	}
}

func loadConfig(throttleOverride string) (config, error) {
	env := func(key string) string {
		return strings.TrimSpace(os.Getenv(key))
	}

	cfg := config{
		ShopifyStore:       env("SHOPIFY_STORE"),
		ShopifyAccessToken: firstNonEmpty(env("SHOPIFY_ADMIN_TOKEN"), env("SHOPIFY_ACCESS_TOKEN")),
		OpenAIKey:          env("OPENAI_API_KEY"),
		Model:              firstNonEmpty(env("MODEL"), defaultModel),
		ModelsCSV:          firstNonEmpty(env("MODELS"), defaultModels),
		ProductsLimit:      defaultProductsLimit,
		Temperature:        defaultTemperature,
		OpenAIWorkers:      defaultOpenAIWorkers,
		MinScore:           defaultMinScore,
	}

	if cfg.ShopifyStore == "" {
		return config{}, errors.New("SHOPIFY_STORE is required")
	}

	if cfg.ShopifyAccessToken == "" {
		return config{}, errors.New("SHOPIFY_ACCESS_TOKEN (or SHOPIFY_ADMIN_TOKEN) is required")
	}

	if cfg.OpenAIKey == "" {
		return config{}, errors.New("OPENAI_API_KEY is required")
	}

	if limitStr := env("PRODUCTS_LIMIT"); limitStr != "" {
		if limit, err := strconv.Atoi(limitStr); err == nil && limit > 0 {
			cfg.ProductsLimit = limit
		} else {
			return config{}, fmt.Errorf("invalid PRODUCTS_LIMIT value %q", limitStr)
		}
	}

	if workersStr := env("OPENAI_WORKERS"); workersStr != "" {
		if workers, err := strconv.Atoi(workersStr); err == nil && workers > 0 {
			cfg.OpenAIWorkers = workers
		} else {
			return config{}, fmt.Errorf("invalid OPENAI_WORKERS value %q", workersStr)
		}
	}

	if tempStr := env("OPENAI_TEMPERATURE"); tempStr != "" {
		if temp, err := strconv.ParseFloat(tempStr, 64); err == nil {
			cfg.Temperature = temp
		} else {
			return config{}, fmt.Errorf("invalid OPENAI_TEMPERATURE value %q", tempStr)
		}
	}

	throttle := time.Duration(defaultThrottleMillis) * time.Millisecond
	if throttleOverride != "" {
		if parsed, err := time.ParseDuration(throttleOverride); err == nil {
			throttle = parsed
		} else {
			return config{}, fmt.Errorf("invalid throttle duration %q", throttleOverride)
		}
	}
	cfg.Throttle = throttle

	cfg.ShopifyStore = sanitizeStoreDomain(cfg.ShopifyStore)

	return cfg, nil
}

func (r *runner) run(ctx context.Context) error {
	log.Printf("Starting alt text generation for store %s (limit %d, model %s, workers=%d, dry=%t, force=%t)",
		r.cfg.ShopifyStore, r.cfg.ProductsLimit, r.cfg.Model, r.cfg.OpenAIWorkers, r.dryRun, r.force)

	var after *string
	totalProducts := 0
	totalImages := 0
	updated := 0
	generated := 0
	skippedExisting := 0

	for {
		page, err := r.shopify.FetchProducts(ctx, r.cfg.ProductsLimit, after)
		if err != nil {
			return fmt.Errorf("fetch products: %w", err)
		}

		if len(page.Products) == 0 && !page.HasNext {
			log.Print("No products found or end of catalog reached.")
			break
		}

		jobs := make([]imageJob, 0)

		for _, product := range page.Products {
			totalProducts++

			// Filter by handle if specified
			if r.cfg.HandleFilter != "" {
				filterHandles := splitCSV(r.cfg.HandleFilter)
				matchesFilter := false
				for _, fh := range filterHandles {
					if strings.Contains(strings.ToLower(product.Handle), strings.ToLower(fh)) {
						matchesFilter = true
						break
					}
				}
				if !matchesFilter {
					continue
				}
			}

			for _, img := range product.Images {
				totalImages++
				if !r.force && img.AltText != "" {
					skippedExisting++
					continue
				}

				jobs = append(jobs, imageJob{
					product: product,
					image:   img,
				})

				// Apply image limit if specified
				if r.cfg.LimitImages > 0 && len(jobs) >= r.cfg.LimitImages {
					break
				}
			}

			// Break outer loop if we've hit the limit
			if r.cfg.LimitImages > 0 && len(jobs) >= r.cfg.LimitImages {
				break
			}
		}

		if len(jobs) > 0 {
			pageGenerated, pageUpdated := r.processAltJobs(ctx, jobs)
			generated += pageGenerated
			updated += pageUpdated
		}

		if !page.HasNext {
			break
		}
		after = &page.EndCursor
		log.Printf("Continuing to next page (after=%s)", page.EndCursor)
	}

	log.Printf("Run complete: products=%d images=%d generated=%d updated=%d skipped_existing=%d",
		totalProducts, totalImages, generated, updated, skippedExisting)
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		t := strings.TrimSpace(p)
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

type FewShot struct {
	Title    string `json:"title"`
	Brand    string `json:"brand"`
	Type     string `json:"type"`
	Alt      string `json:"alt"`
	ImageURL string `json:"image_url,omitempty"`
}

type reportRow struct {
	Time         string   `json:"time"`
	ProductID    string   `json:"product_id"`
	Title        string   `json:"title"`
	Vendor       string   `json:"vendor"`
	Type         string   `json:"type"`
	Handle       string   `json:"handle"`
	ImageURL     string   `json:"image_url"`
	Output       string   `json:"alt"`
	Score        int      `json:"score"`
	Reasons      []string `json:"reasons"`
	WroteShopify bool     `json:"wrote_shopify"`
	ModelTried   []string `json:"models"`
}

var reportMu sync.Mutex

func appendReport(path string, row reportRow) {
	if strings.TrimSpace(path) == "" {
		return
	}
	reportMu.Lock()
	defer reportMu.Unlock()

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("report open err: %v", err)
		return
	}
	defer f.Close()

	enc, _ := json.Marshal(row)
	f.Write(append(enc, '\n'))
}

func loadFewShot(path string) ([]FewShot, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []FewShot
	if strings.HasSuffix(strings.ToLower(path), ".csv") {
		// Simple CSV parser (title,brand,type,alt)
		lines := strings.Split(string(b), "\n")
		for _, ln := range lines {
			ln = strings.TrimSpace(ln)
			if ln == "" || strings.HasPrefix(ln, "#") {
				continue
			}
			cols := strings.Split(ln, ",")
			if len(cols) < 4 {
				continue
			}
			rows = append(rows, FewShot{
				Title: strings.TrimSpace(cols[0]),
				Brand: strings.TrimSpace(cols[1]),
				Type:  strings.TrimSpace(cols[2]),
				Alt:   strings.TrimSpace(cols[3]),
			})
		}
	} else {
		if err := json.Unmarshal(b, &rows); err != nil {
			return nil, err
		}
	}
	return rows, nil
}

func sanitizeStoreDomain(store string) string {
	store = strings.TrimSpace(strings.ToLower(store))
	store = strings.TrimPrefix(store, "https://")
	store = strings.TrimPrefix(store, "http://")
	store = strings.TrimSuffix(store, "/")
	return store
}

// Shopify client + models

type shopifyClient struct {
	store      string
	token      string
	httpClient *http.Client
}

func newShopifyClient(store, token string, httpClient *http.Client) *shopifyClient {
	return &shopifyClient{
		store:      store,
		token:      token,
		httpClient: httpClient,
	}
}

func (c *shopifyClient) graphqlURL() string {
	return fmt.Sprintf("https://%s/admin/api/%s/graphql.json", c.store, shopifyAPIVersion)
}

type product struct {
	ID          string
	Title       string
	Vendor      string
	ProductType string
	Handle      string
	Tags        []string
	Images      []mediaImage
}

type mediaImage struct {
	ID      string
	ImageID string
	URL     string
	AltText string
}

func (m mediaImage) FileID() string {
	// For fileUpdate mutation, use the MediaImage.id directly
	// According to Shopify docs, MediaImage implements the File interface
	return m.ID
}

type imageJob struct {
	product product
	image   mediaImage
}

type productsPage struct {
	Products  []product
	EndCursor string
	HasNext   bool
}

type graphQLRequest struct {
	Query     string                 `json:"query"`
	Variables map[string]interface{} `json:"variables"`
}

type graphQLError struct {
	Message string `json:"message"`
}

func (c *shopifyClient) FetchProducts(ctx context.Context, first int, after *string) (productsPage, error) {
	reqBody := graphQLRequest{
		Query: `query ProductsWithMedia($first: Int!, $after: String) {
  products(first: $first, after: $after, sortKey: ID) {
    edges {
      cursor
      node {
        id
        title
        vendor
        productType
        handle
        tags
        media(first: 100) {
          edges {
            node {
              __typename
              ... on MediaImage {
                id
                image {
                  id
                  url
                  altText
                }
              }
            }
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}`,
		Variables: map[string]interface{}{
			"first": first,
			"after": after,
		},
	}

	if after == nil {
		reqBody.Variables["after"] = nil
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return productsPage{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.graphqlURL(), bytes.NewReader(bodyBytes))
	if err != nil {
		return productsPage{}, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Shopify-Access-Token", c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return productsPage{}, fmt.Errorf("graphql request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return productsPage{}, fmt.Errorf("shopify graphql status %d", resp.StatusCode)
	}

	var decoded struct {
		Data struct {
			Products struct {
				Edges []struct {
					Cursor string `json:"cursor"`
					Node   struct {
						ID          string   `json:"id"`
						Title       string   `json:"title"`
						Vendor      string   `json:"vendor"`
						ProductType string   `json:"productType"`
						Handle      string   `json:"handle"`
						Tags        []string `json:"tags"`
						Media       struct {
							Edges []struct {
								Node struct {
									Type  string `json:"__typename"`
									ID    string `json:"id"`
									Image struct {
										ID      string  `json:"id"`
										URL     string  `json:"url"`
										AltText *string `json:"altText"`
									} `json:"image"`
								} `json:"node"`
							} `json:"edges"`
						} `json:"media"`
					} `json:"node"`
				} `json:"edges"`
				PageInfo struct {
					HasNextPage bool   `json:"hasNextPage"`
					EndCursor   string `json:"endCursor"`
				} `json:"pageInfo"`
			} `json:"products"`
		} `json:"data"`
		Errors []graphQLError `json:"errors"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return productsPage{}, fmt.Errorf("decode response: %w", err)
	}

	if len(decoded.Errors) > 0 {
		var messages []string
		for _, e := range decoded.Errors {
			messages = append(messages, e.Message)
		}
		return productsPage{}, fmt.Errorf("shopify graphql errors: %s", strings.Join(messages, "; "))
	}

	page := productsPage{
		EndCursor: decoded.Data.Products.PageInfo.EndCursor,
		HasNext:   decoded.Data.Products.PageInfo.HasNextPage,
	}

	for _, edge := range decoded.Data.Products.Edges {
		prod := product{
			ID:          edge.Node.ID,
			Title:       edge.Node.Title,
			Vendor:      edge.Node.Vendor,
			ProductType: edge.Node.ProductType,
			Handle:      edge.Node.Handle,
			Tags:        edge.Node.Tags,
		}

		for _, mediaEdge := range edge.Node.Media.Edges {
			if mediaEdge.Node.Type != "MediaImage" {
				continue
			}
			img := mediaImage{
				ID:      mediaEdge.Node.ID,
				ImageID: mediaEdge.Node.Image.ID,
				URL:     mediaEdge.Node.Image.URL,
			}
			if mediaEdge.Node.Image.AltText != nil {
				img.AltText = strings.TrimSpace(*mediaEdge.Node.Image.AltText)
			}
			prod.Images = append(prod.Images, img)
		}

		page.Products = append(page.Products, prod)
	}

	return page, nil
}

type fileUpdateInput struct {
	ID  string `json:"id"`
	Alt string `json:"alt"`
}

func (c *shopifyClient) UpdateAltText(ctx context.Context, update fileUpdateInput) error {
	reqBody := graphQLRequest{
		Query: `mutation fileUpdate($files: [FileUpdateInput!]!) {
  fileUpdate(files: $files) {
    files {
      id
      alt
      fileStatus
    }
    userErrors {
      field
      message
      code
    }
  }
}`,
		Variables: map[string]interface{}{
			"files": []fileUpdateInput{update},
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.graphqlURL(), bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Shopify-Access-Token", c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("shopify update: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("shopify update status %d", resp.StatusCode)
	}

	var decoded struct {
		Data struct {
			FileUpdate struct {
				Files []struct {
					ID         string `json:"id"`
					Alt        string `json:"alt"`
					FileStatus string `json:"fileStatus"`
				} `json:"files"`
				UserErrors []struct {
					Field   []string `json:"field"`
					Message string   `json:"message"`
					Code    string   `json:"code"`
				} `json:"userErrors"`
			} `json:"fileUpdate"`
		} `json:"data"`
		Errors []graphQLError `json:"errors"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	if len(decoded.Errors) > 0 {
		var messages []string
		for _, e := range decoded.Errors {
			messages = append(messages, e.Message)
		}
		return fmt.Errorf("shopify graphql errors: %s", strings.Join(messages, "; "))
	}

	if errs := decoded.Data.FileUpdate.UserErrors; len(errs) > 0 {
		var messages []string
		for _, e := range errs {
			path := strings.Join(e.Field, ".")
			if path != "" {
				messages = append(messages, fmt.Sprintf("%s: %s (%s)", path, e.Message, e.Code))
			} else {
				messages = append(messages, fmt.Sprintf("%s (%s)", e.Message, e.Code))
			}
		}
		return fmt.Errorf("shopify user errors: %s", strings.Join(messages, "; "))
	}

	return nil
}

func (r *runner) processAltJobs(ctx context.Context, jobs []imageJob) (int, int) {
	if len(jobs) == 0 {
		return 0, 0
	}

	type generationResult struct {
		job imageJob
		alt string
		err error
	}

	workerCount := r.cfg.OpenAIWorkers
	if workerCount < 1 {
		workerCount = 1
	}

	jobsCh := make(chan imageJob)
	resultsCh := make(chan generationResult)

	var wg sync.WaitGroup
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobsCh {
				alt, err := r.openai.GenerateAltText(ctx, promptInput{
					ProductTitle: job.product.Title,
					Vendor:       job.product.Vendor,
					ProductType:  job.product.ProductType,
					Handle:       job.product.Handle,
					Tags:         job.product.Tags,
					ImageURL:     job.image.URL,
				})
				select {
				case <-ctx.Done():
					return
				case resultsCh <- generationResult{job: job, alt: alt, err: err}:
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	go func() {
		defer close(jobsCh)
		for _, job := range jobs {
			select {
			case <-ctx.Done():
				return
			case jobsCh <- job:
			}
		}
	}()

	generated := 0
	updated := 0

	for {
		select {
		case <-ctx.Done():
			return generated, updated
		case res, ok := <-resultsCh:
			if !ok {
				return generated, updated
			}

			// Initialize report row
			reportData := reportRow{
				Time:      time.Now().Format(time.RFC3339),
				ProductID: res.job.product.ID,
				Title:     res.job.product.Title,
				Vendor:    res.job.product.Vendor,
				Type:      res.job.product.ProductType,
				Handle:    res.job.product.Handle,
				ImageURL:  res.job.image.URL,
			}

			if res.err != nil {
				log.Printf("OpenAI error for product %s (%s): %v", res.job.product.ID, res.job.image.ID, res.err)
				reportData.Output = ""
				reportData.Reasons = []string{fmt.Sprintf("openai_error: %v", res.err)}
				appendReport(r.cfg.ReportPath, reportData)
				continue
			}

			alt := strings.TrimSpace(res.alt)
			if alt == "" {
				log.Printf("OpenAI returned empty alt for product %s (%s)", res.job.product.ID, res.job.image.ID)
				reportData.Output = ""
				reportData.Reasons = []string{"empty_output"}
				appendReport(r.cfg.ReportPath, reportData)
				continue
			}

			// Evaluate quality
			_, reasons, score := evaluateAltV2(alt, promptInput{
				ProductTitle: res.job.product.Title,
				Vendor:       res.job.product.Vendor,
				ProductType:  res.job.product.ProductType,
				Handle:       res.job.product.Handle,
				Tags:         res.job.product.Tags,
				ImageURL:     res.job.image.URL,
			})

			reportData.Output = alt
			reportData.Score = score
			reportData.Reasons = reasons

			// Gate by min-score
			if score < r.cfg.MinScore {
				log.Printf("SKIP (score %d < %d) %s reasons=%v text=%q",
					score, r.cfg.MinScore, res.job.image.URL, reasons, alt)
				reportData.WroteShopify = false
				appendReport(r.cfg.ReportPath, reportData)
				continue
			}

			generated++
			log.Printf("Generated alt text for %s (score=%d): %q", res.job.image.URL, score, alt)

			if r.dryRun {
				reportData.WroteShopify = false
				appendReport(r.cfg.ReportPath, reportData)
				continue
			}

			if err := r.shopify.UpdateAltText(ctx, fileUpdateInput{ID: res.job.image.FileID(), Alt: alt}); err != nil {
				log.Printf("Shopify update error for image %s: %v", res.job.image.ID, err)
				reportData.WroteShopify = false
				reportData.Reasons = append(reportData.Reasons, fmt.Sprintf("shopify_error: %v", err))
				appendReport(r.cfg.ReportPath, reportData)
				continue
			}
			updated++
			reportData.WroteShopify = true
			appendReport(r.cfg.ReportPath, reportData)

			if r.cfg.Throttle > 0 {
				if err := sleepContext(ctx, r.cfg.Throttle); err != nil {
					return generated, updated
				}
			}
		}
	}
}

func sleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// ==== OpenAI client + prompt (E-E-A-T + SEO tuned) ====

type openAIClient struct {
	apiKey       string
	model        string
	models       []string
	modelCompare bool
	temperature  float64
	fewShots     []FewShot
	httpClient   *http.Client
}

func newOpenAIClient(apiKey, model string, models []string, modelCompare bool, temperature float64, fewShots []FewShot, httpClient *http.Client) *openAIClient {
	return &openAIClient{
		apiKey:       apiKey,
		model:        model,
		models:       models,
		modelCompare: modelCompare,
		temperature:  temperature,
		fewShots:     fewShots,
		httpClient:   httpClient,
	}
}

type promptInput struct {
	ProductTitle string
	Vendor       string
	ProductType  string
	Handle       string
	Tags         []string
	ImageURL     string
}

func (c *openAIClient) GenerateAltText(ctx context.Context, input promptInput) (string, error) {
	if strings.TrimSpace(input.ImageURL) == "" {
		return "", errors.New("missing image URL")
	}

	models := c.models
	if len(models) == 0 {
		models = []string{c.model}
	}

	if c.modelCompare {
		// Run all models and print comparison; return best by score
		var best string
		bestScore := -1
		for _, m := range models {
			out, _, err := c.callWithModel(ctx, input, m, c.temperature)
			if err != nil {
				log.Printf("[compare] %s error: %v", m, err)
				continue
			}
			out = normalizeAlt(out)
			ok, reasons, score := evaluateAltV2(out, input)
			log.Printf("[compare] %s => %q ok=%t score=%d reasons=%v", m, out, ok, score, reasons)
			if score > bestScore {
				best, bestScore = out, score
			}
		}
		if best == "" {
			return "", errors.New("all models failed in --model-compare")
		}
		return best, nil
	}

	// Fallback chain
	for i, m := range models {
		out, raw, err := c.callWithModel(ctx, input, m, c.temperature)
		if err != nil {
			log.Printf("model %s failed (%d/%d): %v", m, i+1, len(models), err)
			continue
		}
		out = normalizeAlt(out)
		ok, reasons, score := evaluateAltV2(out, input)
		if ok {
			return out, nil
		}

		// One deterministic low-temp retry before falling back
		out2, _, err2 := c.callWithModel(ctx, input, m, 0.0)
		if err2 == nil {
			out2 = normalizeAlt(out2)
			ok2, reasons2, _ := evaluateAltV2(out2, input)
			if ok2 {
				return out2, nil
			}
			log.Printf("quality fail on %s (score=%d): %q reasons=%v; retry reasons=%v (raw=%s)",
				m, score, out, reasons, reasons2, truncate(raw, 200))
		} else {
			log.Printf("retry error on %s: %v (raw=%s)", m, err2, truncate(raw, 200))
		}
	}
	return "", errors.New("all models produced low-quality outputs")
}

func (c *openAIClient) callWithModel(ctx context.Context, input promptInput, model string, temp float64) (string, string, error) {
	payload := c.buildOpenAIPayload(model, temp, input)
	return c.callResponses(ctx, payload)
}

func (c *openAIClient) buildOpenAIPayload(model string, temp float64, input promptInput) openAIRequest {
	sys := `You are an accessibility + SEO specialist for e-commerce images.
When describing, be *visually grounded* and concise:
- State exact product name and brand once, naturally.
- Mention ONE clearly visible trait from the image: container (bottle, drum, pail, jug, tote), apparent size, color, label/markings, cap style.
- Avoid hype, purity/biocidal claims, emojis, ALL-CAPS. One sentence, 12–18 words, no trailing period.
If uncertain about a trait, omit it; never invent details.`

	// Domain context
	domain := ""
	if input.ProductType != "" {
		domain = "Product category: " + input.ProductType
	}

	// Add relevant tags
	tagInfo := ""
	if len(input.Tags) > 0 {
		relevantTags := extractRelevantTags(input.Tags)
		if len(relevantTags) > 0 {
			tagInfo = "Product tags: " + strings.Join(relevantTags, ", ")
		}
	}

	usr := strings.Join([]string{
		"Product title: " + input.ProductTitle,
		"Brand/vendor: " + input.Vendor,
		domain,
		"Handle: " + input.Handle,
		tagInfo,
		"Instructions:",
		"- Return only the alt text line.",
		"- 12–18 words; sentence case.",
		"- For chemical/industrial products, prioritize container type and size over subjective descriptions.",
	}, "\n")

	messages := []openAIMessage{
		{Role: "system", Content: []openAIContent{{Type: "text", Text: sys}}},
	}

	// Inject few-shot examples if available
	if len(c.fewShots) > 0 {
		var sb strings.Builder
		sb.WriteString("Here are style examples:\n")
		limit := 3
		if len(c.fewShots) < limit {
			limit = len(c.fewShots)
		}
		for i := 0; i < limit; i++ {
			f := c.fewShots[i]
			fmt.Fprintf(&sb, "- %s / %s → %s\n", f.Title, f.Brand, f.Alt)
		}
		messages = append(messages, openAIMessage{
			Role:    "system",
			Content: []openAIContent{{Type: "text", Text: sb.String()}},
		})
	}

	// Main user request with image
	messages = append(messages, openAIMessage{
		Role: "user",
		Content: []openAIContent{
			{Type: "text", Text: usr},
			{Type: "image_url", ImageURL: &openAIImageURL{URL: input.ImageURL}},
		},
	})

	req := openAIRequest{
		Model:     model,
		MaxTokens: maxOutputTokens,
		Input:     messages,
	}
	// Note: gpt-5-nano doesn't support temperature parameter
	// Only include if non-default and model supports it
	// if temp != defaultTemperature {
	// 	req.Temperature = temp
	// }
	return req
}

func normalizeAlt(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 0 && strings.HasSuffix(s, ".") {
		trimmed := strings.TrimSuffix(s, ".")
		if !strings.Contains(trimmed, ".") {
			s = trimmed
		}
	}
	return s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_output_tokens,omitempty"`
	Input       []openAIMessage `json:"input"`
}

type openAIMessage struct {
	Role    string          `json:"role"`
	Content []openAIContent `json:"content"`
}

type openAIImageURL struct {
	URL string `json:"url"`
}

type openAIContent struct {
	Type     string           `json:"type"`                    // "text" | "image_url"
	Text     string           `json:"text,omitempty"`          // when Type == "text"
	ImageURL *openAIImageURL  `json:"image_url,omitempty"`     // when Type == "image_url"
}

type openAIResponse struct {
	Output []struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
}

type openAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   any    `json:"param"`
	} `json:"error"`
}

func (c *openAIClient) callResponses(ctx context.Context, reqBody openAIRequest) (string, string, error) {
	// Convert to Chat Completions API format
	chatReq := map[string]interface{}{
		"model":      reqBody.Model,
		"messages":   reqBody.Input,
		"max_tokens": reqBody.MaxTokens,
	}

	bodyBytes, err := json.Marshal(chatReq)
	if err != nil {
		return "", "", fmt.Errorf("marshal openai request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", "", fmt.Errorf("new openai request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("openai request: %w", err)
	}
	defer resp.Body.Close()

	respBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		var apiErr openAIErrorResponse
		if err := json.Unmarshal(respBytes, &apiErr); err == nil && apiErr.Error.Message != "" {
			return "", string(respBytes), fmt.Errorf("openai error (%d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return "", string(respBytes), fmt.Errorf("openai status %d", resp.StatusCode)
	}

	// Parse Chat Completions response format
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBytes, &decoded); err != nil {
		return "", string(respBytes), fmt.Errorf("decode openai response: %w", err)
	}

	if len(decoded.Choices) > 0 && decoded.Choices[0].Message.Content != "" {
		return decoded.Choices[0].Message.Content, string(respBytes), nil
	}
	return "", string(respBytes), nil
}

// ===== Alt text quality evaluator (SEO + E-E-A-T guardrails) =====

var hypeWords = []string{"premium", "best", "top", "amazing", "cheap", "world-class", "leading"}
var biocideWords = []string{"disinfect", "disinfection", "antibacterial", "antimicrobial", "sanitiz", "steriliz"}

// Visual detail indicators - words that show the AI actually described what it saw
var visualDetailWords = []string{
	// Container types
	"drum", "bottle", "jar", "jug", "pail", "tote", "bag", "pouch", "can", "tube", "container", "vial",
	// Colors
	"white", "black", "blue", "red", "green", "yellow", "amber", "clear", "brown", "gray",
	// Materials
	"plastic", "glass", "metal", "steel", "hdpe", "aluminum",
	// Features
	"label", "cap", "lid", "pump", "dispenser", "valve", "handle", "spout",
	// Sizes (with numbers)
	"gallon", "liter", "ml", "oz", "quart", "pint", "kg", "lb",
}

// Generic phrases that show lack of specificity
var genericPhrases = []string{
	"product image", "product photo", "item shown", "as pictured",
	"high quality", "great value", "excellent product",
}

var nonAlpha = regexp.MustCompile(`[^A-Za-z0-9]+`)

// evaluateAlt is the legacy boolean evaluator - kept for compatibility
func evaluateAlt(alt string, in promptInput) (bool, []string) {
	ok, reasons, _ := evaluateAltV2(alt, in)
	return ok, reasons
}

// evaluateAltV2 returns (ok, reasons, score 0-100)
func evaluateAltV2(alt string, in promptInput) (bool, []string, int) {
	score := 100
	var reasons []string

	wc := len(strings.Fields(alt))
	if wc < 12 || wc > 18 {
		score -= 25
		reasons = append(reasons, fmt.Sprintf("bad length (%d words)", wc))
	}

	// Title keyword coverage
	want := significantTokens(in.ProductTitle, 5)
	got := 0
	for _, t := range want {
		if containsWord(alt, t) {
			got++
		}
	}
	if got < min(2, len(want)) {
		score -= 15
		reasons = append(reasons, "weak title coverage")
	}

	// Brand coverage
	if in.Vendor != "" && !containsWord(alt, in.Vendor) {
		score -= 10
		reasons = append(reasons, "missing brand")
	}

	// Penalize generic phrases
	if containsAnyFold(alt, genericPhrases) {
		score -= 20
		reasons = append(reasons, "generic phrasing")
	}

	// Visual detail presence (reward image-specific descriptions)
	if !containsAnyFold(alt, visualDetailWords) {
		score -= 10
		reasons = append(reasons, "no visual detail")
	}

	// Hype and biocidal claims
	if containsAnyFold(alt, hypeWords) {
		score -= 20
		reasons = append(reasons, "hype")
	}
	if containsAnyFold(alt, biocideWords) {
		score -= 30
		reasons = append(reasons, "biocidal claim")
	}

	// Repetition
	if repetitionScore(alt) > 2 {
		score -= 10
		reasons = append(reasons, "repetition")
	}

	// ALL-CAPS words
	if hasAllCapsWord(alt) {
		score -= 5
		reasons = append(reasons, "ALL-CAPS")
	}

	if score < 70 {
		return false, reasons, score
	}
	return true, reasons, score
}

func hasAllCapsWord(s string) bool {
	for _, tok := range strings.Fields(s) {
		if len(tok) >= 4 && isAllCaps(tok) {
			return true
		}
	}
	return false
}

func significantTokens(s string, max int) []string {
	lower := strings.ToLower(s)
	lower = strings.ReplaceAll(lower, "/", " ")
	lower = nonAlpha.ReplaceAllString(lower, " ")
	words := strings.Fields(lower)
	stopwords := map[string]bool{"the": true, "and": true, "for": true, "with": true, "of": true, "a": true, "an": true, "by": true, "to": true, "in": true, "on": true}
	var tokens []string
	for _, w := range words {
		if stopwords[w] || len(w) <= 2 {
			continue
		}
		tokens = append(tokens, w)
		if len(tokens) >= max {
			break
		}
	}
	return tokens
}

func containsWord(s, w string) bool {
	pattern := regexp.MustCompile(`\b` + regexp.QuoteMeta(strings.ToLower(w)) + `\b`)
	return pattern.FindStringIndex(strings.ToLower(s)) != nil
}

func containsAnyFold(s string, list []string) bool {
	lower := strings.ToLower(s)
	for _, candidate := range list {
		if strings.Contains(lower, strings.ToLower(candidate)) {
			return true
		}
	}
	return false
}

func repetitionScore(s string) int {
	counts := make(map[string]int)
	for _, word := range strings.Fields(strings.ToLower(s)) {
		counts[word]++
	}
	maxSeen := 0
	for _, c := range counts {
		if c > maxSeen {
			maxSeen = c
		}
	}
	return maxSeen
}

func isAllCaps(tok string) bool {
	hasLetter := false
	for _, r := range tok {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
		}
	}
	return hasLetter
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// extractRelevantTags filters tags to keep only those that provide useful context
// (sizes, certifications, forms, hazard classes) and excludes generic marketing tags
func extractRelevantTags(tags []string) []string {
	var relevant []string
	genericTags := map[string]bool{
		"new": true, "sale": true, "featured": true, "popular": true,
		"bestseller": true, "trending": true, "hot": true,
	}

	// Patterns that indicate relevant tags for industrial/lab chemicals
	relevantPatterns := []string{
		"gallon", "liter", "drum", "pail", "quart", "pint", "ml", "oz",
		"acs", "usp", "grade", "certified", "pure", "technical",
		"liquid", "solid", "powder", "pellets", "flake",
		"hazard", "corrosive", "flammable", "oxidizer",
	}

	for _, tag := range tags {
		tagLower := strings.ToLower(strings.TrimSpace(tag))
		if tagLower == "" || genericTags[tagLower] {
			continue
		}

		// Check if tag contains any relevant pattern
		for _, pattern := range relevantPatterns {
			if strings.Contains(tagLower, pattern) {
				relevant = append(relevant, tag)
				break
			}
		}

		// Limit to first 3 relevant tags to avoid clutter
		if len(relevant) >= 3 {
			break
		}
	}

	return relevant
}
