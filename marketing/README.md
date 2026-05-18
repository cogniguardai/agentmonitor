# marketing/

Source for the marketing site at <https://cogniguardai.com/>.

Static, zero-build, single-file landing page. No frameworks, no CDN
dependencies. The whole site is `index.html` plus a few image assets.

## Files

- `index.html` &mdash; the entire site. Inline CSS + one small `<script>`
  for the "copy install command" button.
- `og.svg` / `og.png` &mdash; 1200x630 Open Graph card. SVG is the source;
  PNG is the rasterised version social-media scrapers actually use.
- `screenshot-welcome.png` / `screenshot-detonations.png` &mdash; the
  in-app dashboard screenshots embedded in the page (captured via
  Playwright from a running dev server).

## Local preview

From the repo root:

```bash
python -m http.server 8090 --directory marketing
# then open http://localhost:8090
```

## Deployment

The site is deployed via [Cloudflare Pages](https://pages.cloudflare.com/)
with the build output directory set to `marketing/`. Every push to `main`
triggers an automatic deploy.

## License

Same as the rest of this repository: MIT.
