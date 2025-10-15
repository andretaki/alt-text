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
MODEL=gpt-5-nano-2025-08-07   # or gpt-5-mini-2025-08-07
PRODUCTS_LIMIT=50             # products per page
# OPENAI_TEMPERATURE=0.2      # optional; defaults to 0.2
# OPENAI_WORKERS=4            # parallel OpenAI calls (>=1)
```

Set up dependencies and build:

```bash
go mod tidy
go build -o altgen
```

## 3. Running

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

Tune the per-request delay (default `750ms`) to respect Shopify throttles:

```bash
./altgen --throttle=1s
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
- Avoid speculation, purity/disinfection claims, all-caps, or promotional hype.
- Return only the alt text line (no trailing punctuation or extra sentences).
- Quality gate enforces 12–18 words, brand inclusion, visible trait mention, and rejects hype or biocidal claims (auto-retries once if needed).

Adjust these rules in code as needed.

## 6. Shopify Permissions & Scopes

- `fileUpdate` requires `write_files` (or `write_themes`) and appropriate user permissions.
- Files must be in `READY` state before updates apply; pending files may be delayed.
- `productUpdateMedia` is deprecated—stick to `fileUpdate`.
- Reference: Shopify scopes overview.

## 7. Useful GraphQL Snippets

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

## 8. Error Handling

- `401`/`403`: confirm Admin token, app install status, and `write_files` scope.
- `userErrors` from `fileUpdate`: inspect returned `field`, `message`, and `code`.
- `fileStatus` of `pending`/`processing`: retry once status switches to `READY`.
- Variant mapping: variant-specific alt text may require joining `variant.image.id` to `MediaImage`.
- OpenAI Errors: ensure the Responses API is called correctly and parse `output[0].content[0].text`.

## 9. Operational Tips

- Control costs with `gpt-5-nano`; set scheduled runs with product limits.
- Default run skips existing ALT text; use `--force` for idempotent overwrites.
- Log output (e.g., pipe stdout) or wrap with process supervisors.
- Schedule via cron or GitHub Actions; only the three env vars are required.
- Optional: export current media ALT text via GraphQL before first run.
- Increase `OPENAI_WORKERS` to parallelise image processing; decrease if you see OpenAI rate limits.

## 10. Extending the Tool

- Add filters (`--collection`, `--handle`) to limit scope.
- Emit CSV/JSON reports per product/image/ALT text.
- Add heuristics (minimum word count, trailing punctuation cleanup).
- Swap to the official OpenAI Go SDK when you want SDK ergonomics.

## 11. Security Notes

- Keep `SHOPIFY_ADMIN_TOKEN` and `OPENAI_API_KEY` in `.env`; never commit them.
- In CI, manage credentials via encrypted secrets.
- Scope the custom app strictly to required permissions.

## 12. References

- Shopify `fileUpdate` mutation documentation (scope, examples, status, errors).
- Shopify `MediaImage` object reference.
- Shopify access scopes overview.
- OpenAI Responses API reference.
- OpenAI Go SDK packages.
