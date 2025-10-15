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
	defaultProductsLimit  = 50
	defaultThrottleMillis = 750
	defaultTemperature    = 0.2
	maxOutputTokens       = 180
	defaultOpenAIWorkers  = 4
)

type config struct {
	ShopifyStore       string
	ShopifyAccessToken string
	OpenAIKey          string
	Model              string
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
		dryRunFlag  = flag.Bool("dry", false, "preview changes without writing to Shopify")
		forceFlag   = flag.Bool("force", false, "overwrite existing alt text")
		throttleStr = flag.String("throttle", fmt.Sprintf("%dms", defaultThrottleMillis), "per-request delay (e.g., 500ms, 1s)")
	)
	flag.Parse()

	cfg, err := loadConfig(*throttleStr)
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	httpClient := &http.Client{Timeout: 60 * time.Second}
	shopify := newShopifyClient(cfg.ShopifyStore, cfg.ShopifyAccessToken, httpClient)
	openai := newOpenAIClient(cfg.OpenAIKey, cfg.Model, cfg.Temperature, httpClient)

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
		ProductsLimit:      defaultProductsLimit,
		Temperature:        defaultTemperature,
		OpenAIWorkers:      defaultOpenAIWorkers,
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
	Images      []mediaImage
}

type mediaImage struct {
	ID      string
	ImageID string
	URL     string
	AltText string
}

func (m mediaImage) FileID() string {
	if m.ImageID != "" {
		return m.ImageID
	}
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
						ID          string `json:"id"`
						Title       string `json:"title"`
						Vendor      string `json:"vendor"`
						ProductType string `json:"productType"`
						Handle      string `json:"handle"`
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

			if res.err != nil {
				log.Printf("OpenAI error for product %s (%s): %v", res.job.product.ID, res.job.image.ID, res.err)
				continue
			}

			alt := strings.TrimSpace(res.alt)
			if alt == "" {
				log.Printf("OpenAI returned empty alt for product %s (%s)", res.job.product.ID, res.job.image.ID)
				continue
			}

			generated++
			log.Printf("Generated alt text for %s: %q", res.job.image.URL, alt)

			if r.dryRun {
				continue
			}

			if err := r.shopify.UpdateAltText(ctx, fileUpdateInput{ID: res.job.image.FileID(), Alt: alt}); err != nil {
				log.Printf("Shopify update error for image %s: %v", res.job.image.ID, err)
				continue
			}
			updated++

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
	apiKey      string
	model       string
	temperature float64
	httpClient  *http.Client
}

func newOpenAIClient(apiKey, model string, temperature float64, httpClient *http.Client) *openAIClient {
	return &openAIClient{
		apiKey:      apiKey,
		model:       model,
		temperature: temperature,
		httpClient:  httpClient,
	}
}

type promptInput struct {
	ProductTitle string
	Vendor       string
	ProductType  string
	Handle       string
	ImageURL     string
}

func (c *openAIClient) GenerateAltText(ctx context.Context, input promptInput) (string, error) {
	if strings.TrimSpace(input.ImageURL) == "" {
		return "", errors.New("missing image URL")
	}

	systemPrompt := strings.TrimSpace(`You are an e-commerce accessibility + SEO specialist.
Write ALT text that:
- Maximizes clarity and trust (E-E-A-T) by naming the product precisely and describing ONE concrete visible feature (color/shape/container/label text, etc.).
- Uses natural, factual language; NO hype words (premium, best, cheap, amazing), NO health/biocidal claims.
- Fits accessibility norms: one concise sentence, no emojis, no ALL-CAPS, no extra punctuation.
- 12–18 words total. Return ONLY the alt text line.`)

	var userLines []string
	userLines = append(userLines, fmt.Sprintf("Product title: %s", input.ProductTitle))
	if input.Vendor != "" {
		userLines = append(userLines, fmt.Sprintf("Brand or vendor: %s", input.Vendor))
	}
	if input.ProductType != "" {
		userLines = append(userLines, fmt.Sprintf("Product type: %s", input.ProductType))
	}
	if input.Handle != "" {
		userLines = append(userLines, fmt.Sprintf("Product handle: %s", input.Handle))
	}
	userLines = append(userLines,
		"Instructions:",
		"- Include the exact product name and brand (if provided) naturally (no stuffing).",
		"- Mention ONE visible attribute that is clearly in the image (e.g., container size, color, label).",
		"- 12–18 words; sentence case; avoid trailing period.",
		"- Return only the alt text line.",
	)

	payload := openAIRequest{
		Model:       c.model,
		Temperature: c.temperature,
		MaxTokens:   maxOutputTokens,
		Messages: []openAIMessage{
			{
				Role: "system",
				Content: []openAIContent{
					{Type: "input_text", Text: systemPrompt},
				},
			},
			{
				Role: "user",
				Content: []openAIContent{
					{Type: "input_text", Text: strings.Join(userLines, "\n")},
					{Type: "input_image", ImageURL: input.ImageURL},
				},
			},
		},
	}

	candidate, raw, err := c.callResponses(ctx, payload)
	if err != nil {
		return "", err
	}
	candidate = normalizeAlt(candidate)
	ok, _ := evaluateAlt(candidate, input)
	if !ok {
		payload.Temperature = 0.0
		payload.Messages[1].Content[0].Text = strings.Join([]string{
			strings.Join(userLines, "\n"),
			"Strict rules:",
			"- Must be 12–18 words.",
			"- Must include brand/vendor (if provided) and exact product name once.",
			"- Include one clearly visible trait (e.g., container type/size, color, label text).",
			"- No hype words, no biocidal claims, no ALL-CAPS, no trailing period.",
			"- Return only the alt text line.",
		}, "\n")

		candidate2, _, err2 := c.callResponses(ctx, payload)
		if err2 != nil {
			return "", fmt.Errorf("openai retry failed: %w", err2)
		}
		candidate2 = normalizeAlt(candidate2)
		ok2, reasons := evaluateAlt(candidate2, input)
		if !ok2 {
			return "", fmt.Errorf("openai returned low-quality alt text after retry: %q (raw=%s) reasons: %v", candidate2, truncate(raw, 220), reasons)
		}
		return candidate2, nil
	}
	return candidate, nil
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
	Messages    []openAIMessage `json:"messages"`
}

type openAIMessage struct {
	Role    string          `json:"role"`
	Content []openAIContent `json:"content"`
}

type openAIContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
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
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", "", fmt.Errorf("marshal openai request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/responses", bytes.NewReader(bodyBytes))
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

	var decoded openAIResponse
	if err := json.Unmarshal(respBytes, &decoded); err != nil {
		return "", string(respBytes), fmt.Errorf("decode openai response: %w", err)
	}

	for _, out := range decoded.Output {
		for _, c := range out.Content {
			if strings.TrimSpace(c.Text) != "" {
				return c.Text, string(respBytes), nil
			}
		}
	}
	return "", string(respBytes), nil
}

// ===== Alt text quality evaluator (SEO + E-E-A-T guardrails) =====

var hypeWords = []string{"premium", "best", "top", "amazing", "cheap", "world-class", "leading"}
var biocideWords = []string{"disinfect", "disinfection", "antibacterial", "antimicrobial", "sanitiz", "steriliz"}

var nonAlpha = regexp.MustCompile(`[^A-Za-z0-9]+`)

func evaluateAlt(alt string, in promptInput) (bool, []string) {
	var issues []string
	wordCount := len(strings.Fields(alt))
	if wordCount < 12 || wordCount > 18 {
		issues = append(issues, fmt.Sprintf("word_count=%d (want 12–18)", wordCount))
	}

	titleSig := significantTokens(in.ProductTitle, 5)
	keywordMatches := 0
	for _, token := range titleSig {
		if containsWord(alt, token) {
			keywordMatches++
		}
	}
	requiredMatches := min(2, len(titleSig))
	if keywordMatches < requiredMatches {
		issues = append(issues, "not enough product keywords from title")
	}

	if in.Vendor != "" && !containsWord(alt, in.Vendor) {
		issues = append(issues, "missing brand/vendor")
	}

	if containsAnyFold(alt, hypeWords) {
		issues = append(issues, "contains hype words")
	}
	if containsAnyFold(alt, biocideWords) {
		issues = append(issues, "contains disinfection/biocidal wording")
	}

	for _, tok := range strings.Fields(alt) {
		if len(tok) >= 4 && isAllCaps(tok) {
			issues = append(issues, "contains ALL-CAPS token")
			break
		}
	}

	if repetitionScore(alt) > 2 {
		issues = append(issues, "excessive repetition")
	}

	return len(issues) == 0, issues
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
