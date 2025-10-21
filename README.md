# Alt Text Generator for Shopify

Generate authoritative, SEO-forward ALT text for every product image on your Shopify store using Go and the OpenAI Responses API. The tool paginates through your entire product catalog, skips media that already have ALT text (unless overridden), and updates Shopify via the supported `fileUpdate` mutation.

## 1. Prerequisites

**Shopify**
- Create a custom app in your store admin (`Settings → Apps and sales channels → Develop apps`) and install it.
- Minimum Admin API access scopes: `read_products` (list products & media) and `write_files` (or `write_themes` for `fileUpdate`).
- Copy the Admin API access token after install.

**OpenAI**
- Generate an API key in the OpenAI dashboard.
- This tool targets the Responses API (recommended for new builds). Reference: OpenAI API docs & Go SDK.

**Local Environment**
- Go 1.22 or newer.
- Internet access from the machine/container that will run the tool.

## 2. Install & Configure

Repository contents:
- `main.go` – executable entry point.
- `go.mod` – module definition and dependencies.
- `.env` – create this file with your secrets (never commit it).

Example `.env`:
```
SHOPIFY_STORE=yourstore.myshopify.com
SHOPIFY_ADMIN_TOKEN=shpat_xxx
OPENAI_API_KEY=sk-xxx

# Model strategy for industrial/lab chemicals (quality-first):
MODEL=gpt-4o-mini                                            # primary: best vision for small text, labels, symbols
MODELS=gpt-4o-mini,gpt-5-mini-2025-08-07,gpt-5-nano-2025-08-07  # fallback chain

# For cost-optimized runs:
# MODEL=gpt-5-mini-2025-08-07
# MODELS=gpt-5-mini-2025-08-07,gpt-5-nano-2025-08-07

PRODUCTS_LIMIT=50                     # products per page
# OPENAI_WORKERS=4                    # parallel OpenAI calls (>=1)
```

Set up dependencies and build:

```bash
go mod tidy
go build -o altgen
```

## 3. Running

**Basic Usage:**

Dry run (prints proposed ALT text without mutating Shopify):
```bash
source .env
./altgen --dry
```

Write ALT text to Shopify:
```bash
./altgen
```

Overwrite images that already have ALT text:
```bash
./altgen --force
```

**Advanced Options:**

Multi-model fallback with quality reporting:
```bash
./altgen --models="gpt-5-nano-2025-08-07,gpt-5-mini-2025-08-07" --min-score=75 --report=run.jsonl
```

Compare multiple models side-by-side (dry-run):
```bash
./altgen --model-compare --dry --models="gpt-5-nano-2025-08-07,gpt-5-mini-2025-08-07,gpt-4o-mini"
```

Use few-shot examples for consistent style:
```bash
./altgen --examples=examples.json --dry
```

Filter by product handles (for targeted updates):
```bash
./altgen --handles="acetone,isopropyl" --dry
```

Test with limited images:
```bash
./altgen --limit-images=10 --dry --report=test.jsonl
```

Tune the per-request delay (default `750ms`) to respect Shopify throttles:
```bash
./altgen --throttle=1s
```

**Complete Command Reference:**
```
--dry                   Preview changes without writing to Shopify
--force                 Overwrite existing alt text
--models=LIST           Comma-separated model fallbacks (first wins)
--model-compare         Generate with all models and print comparison
--examples=PATH         Path to JSON/CSV with few-shot examples
--min-score=N           Minimum quality score to write (0-100, default 70)
--report=PATH           Write JSONL audit trail to file
--handles=LIST          Filter products by handle substrings
--limit-images=N        Max images to process (for testing)
--throttle=DURATION     Per-request delay (e.g., 500ms, 1s)
```

## 4. What It Does

- Paginates products in batches (`PRODUCTS_LIMIT` per page) until the catalog is exhausted.
- For each `MediaImage` on Shopify CDN, if `altText` is empty (or `--force`), it:
  - Sends the product metadata plus the live image URL to the OpenAI Responses API (default: `gpt-5-nano`).
  - Requests authoritative, keyword-rich copy that still follows accessibility best practices (no hype, single sentence).
  - Updates Shopify via `fileUpdate(files:[{id, alt}])`.
- Fan-outs OpenAI requests with a worker pool (`OPENAI_WORKERS`) to maximise throughput.
- Applies a configurable throttle between Shopify writes to stay within Admin API limits.
- Uses an internal quality gate (word count, brand coverage, hypeless copy) with an automatic retry for weak generations.

## 5. SEO & Accessibility Rules

Built into the prompt sent to the model:
- 12–18 words, authoritative yet conversational.
- Mention the brand/vendor when provided in Shopify metadata.
- Call out visually obvious features using natural, factual language grounded in the title and image.
- For chemical/industrial products: prioritize container type (drum, pail, jug, bottle), size, and material over subjective descriptions.
- Avoid speculation, purity/disinfection claims, all-caps, or promotional hype.
- Return only the alt text line (no trailing punctuation or extra sentences).

## 6. Quality Scoring (0-100)

Every generated alt text receives a quality score:

**Scoring breakdown:**
- Start at 100 points
- **-25 points:** Wrong word count (not 12-18 words)
- **-15 points:** Weak title coverage (missing key product terms)
- **-10 points:** Missing brand/vendor name
- **-20 points:** Generic phrases ("product image", "stock photo", etc.)
- **-10 points:** No visual details (container type, color, size, etc.)
- **-20 points:** Hype words ("premium", "best", "amazing", etc.)
- **-30 points:** Biocidal claims ("disinfect", "antibacterial", etc.)
- **-10 points:** Excessive repetition
- **-5 points:** ALL-CAPS words

**Minimum score threshold:** Default is 70. Use `--min-score` to adjust.

## 7. Few-Shot Examples

Provide examples to guide style consistency. Create `examples.json`:

```json
[
  {
    "title": "Acetone ACS Grade 55 Gallon Drum",
    "brand": "ChemCentral",
    "type": "Solvents",
    "alt": "ChemCentral acetone ACS grade solvent in white 55-gallon steel drum with hazard labels"
  },
  {
    "title": "Hydrochloric Acid 37% 2.5L",
    "brand": "LabChem",
    "type": "Acids",
    "alt": "LabChem hydrochloric acid 37% in amber 2.5 liter glass bottle with safety cap"
  }
]
```

Run with: `./altgen --examples=examples.json`

CSV format is also supported (columns: title,brand,type,alt):
```csv
Acetone ACS Grade 55 Gallon,ChemCentral,Solvents,ChemCentral acetone ACS grade solvent in white 55-gallon steel drum
Hydrochloric Acid 37%,LabChem,Acids,LabChem hydrochloric acid 37% in amber 2.5 liter glass bottle
```

Adjust these rules in code as needed.

## 8. JSONL Audit Reports

Track every generation with `--report=PATH`:

```bash
./altgen --report=run.jsonl --min-score=75
```

Each line is a JSON object with:
```json
{
  "time": "2025-10-21T12:34:56Z",
  "product_id": "gid://shopify/Product/123",
  "title": "Acetone ACS Grade 55 Gallon",
  "vendor": "ChemCentral",
  "type": "Solvents",
  "handle": "acetone-acs-55gal",
  "image_url": "https://cdn.shopify.com/...",
  "alt": "ChemCentral acetone ACS grade in white 55-gallon steel drum with hazard labels",
  "score": 95,
  "reasons": [],
  "wrote_shopify": true,
  "models": ["gpt-5-nano-2025-08-07"]
}
```

**Use cases:**
- Quality analysis: `jq '.score' run.jsonl | sort -n | uniq -c`
- Failed generations: `jq 'select(.score < 70)' run.jsonl`
- Model performance: `jq -r '.models[0]' run.jsonl | sort | uniq -c`

## 9. Multi-Model Fallback

Automatic failover for reliability:

```bash
./altgen --models="gpt-4o-mini,gpt-5-mini-2025-08-07,gpt-5-nano-2025-08-07"
```

**Behavior:**
1. Try primary model (first in list)
2. If API error → try next model
3. If low quality score → retry with temp=0.0
4. If still low quality → try next model
5. Continue until success or all models exhausted

**Model comparison mode:**
```bash
./altgen --model-compare --dry --models="gpt-4o-mini,gpt-5-mini-2025-08-07,gpt-5-nano-2025-08-07"
```
Runs all models in parallel, logs scores, returns best result.

### Model Selection Guide for Chemical Products

**For industrial/lab chemicals (recommended):**
```
gpt-4o-mini → gpt-5-mini → gpt-5-nano
```

**Why gpt-4o-mini first for chemicals:**
- ✅ **Sharper vision** on small text: percentages ("37%", "99.9%"), grades ("ACS", "USP", "Tech")
- ✅ **Better symbol recognition**: UN/NFPA diamonds, hazard pictograms, CAS numbers
- ✅ **Container detail accuracy**: distinguishes 55-gal steel drum vs HDPE pail, F-style can vs jug
- ✅ **Grounded descriptions**: anchors to visible features (ribbed grip, bolt ring, cap style), fewer hallucinations
- ✅ **Higher first-pass rate**: consistent 12-18 word outputs, fewer retries

**gpt-5-mini (middle tier):**
- Good balance of vision quality and cost
- Solid for general containers but may miss fine label text

**gpt-5-nano (cost fallback):**
- Cheapest option but tends to miss:
  - Small percentage indicators
  - Grade/certification marks
  - Safety symbols and fine print
- Best used as last-resort fallback only

**Cost vs Quality:**
| Model | Input $/1M | Output $/1M | Best For |
|-------|-----------|-------------|----------|
| gpt-4o-mini | $0.15 | $0.60 | Chemicals with labels, symbols, specs |
| gpt-5-mini | $0.25 | $1.00 | General products, lower detail needs |
| gpt-5-nano | $0.05 | $0.40 | Simple products, high volume, tight budget |

## 10. Shopify Permissions & Scopes

- `fileUpdate` requires `write_files` (or `write_themes`) and appropriate user permissions.
- Files must be in `READY` state before updates apply; pending files may be delayed.
- `productUpdateMedia` is deprecated—stick to `fileUpdate`.
- Reference: Shopify scopes overview.

## 11. Useful GraphQL Snippets

List products with media:

```graphql
query ProductsWithMedia($first: Int!, $after: String) {
  products(first: $first, after: $after, sortKey: ID) {
    edges {
      cursor
      node {
        id
        title
        media(first: 50) {
          edges {
            node {
              __typename
              ... on MediaImage {
                image { id url altText }
              }
            }
          }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
```

Update a single image’s ALT text:

```graphql
mutation fileUpdate($files: [FileUpdateInput!]!) {
  fileUpdate(files: $files) {
    files { id alt fileStatus }
    userErrors { field message code }
  }
}
```

Example variables:

```json
{ "files": [ { "id": "gid://shopify/MediaImage/123456789", "alt": "Short factual alt text" } ] }
```

## 12. Error Handling

- `401`/`403`: confirm Admin token, app install status, and `write_files` scope.
- `userErrors` from `fileUpdate`: inspect returned `field`, `message`, and `code`.
- `fileStatus` of `pending`/`processing`: retry once status switches to `READY`.
- Variant mapping: variant-specific alt text may require joining `variant.image.id` to `MediaImage`.
- OpenAI Errors: ensure the Responses API is called correctly and parse `output[0].content[0].text`.

## 13. Operational Tips

- Control costs with `gpt-5-nano`; set scheduled runs with product limits.
- Default run skips existing ALT text; use `--force` for idempotent overwrites.
- Log output (e.g., pipe stdout) or wrap with process supervisors.
- Schedule via cron or GitHub Actions; only the three env vars are required.
- Optional: export current media ALT text via GraphQL before first run.
- Increase `OPENAI_WORKERS` to parallelise image processing; decrease if you see OpenAI rate limits.
- Use `--limit-images` for smoke tests before full catalog runs.
- Enable `--report` to track quality trends and identify prompt tuning opportunities.
- Compare models with `--model-compare` to find the best cost/quality balance.

## 14. Extending the Tool

- Add filters (`--collection`, `--handle`) to limit scope.
- Emit CSV/JSON reports per product/image/ALT text.
- Add heuristics (minimum word count, trailing punctuation cleanup).
- Swap to the official OpenAI Go SDK when you want SDK ergonomics.
- Variant-specific alt text (detect color/size variations from Shopify variants).
- HTTP retry logic for 429/5xx errors.

## 15. Security Notes

- Keep `SHOPIFY_ADMIN_TOKEN` and `OPENAI_API_KEY` in `.env`; never commit them.
- In CI, manage credentials via encrypted secrets.
- Scope the custom app strictly to required permissions.
- Review JSONL reports for any leaked sensitive data before sharing.

## 16. References

- Shopify `fileUpdate` mutation documentation (scope, examples, status, errors).
- Shopify `MediaImage` object reference.
- Shopify access scopes overview.
- OpenAI Responses API reference.
- OpenAI Go SDK packages.
