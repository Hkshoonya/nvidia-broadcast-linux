# Auto-updating Sponsors — Setup & Reuse

This repo auto-refreshes its sponsor walls (in `README.md` and `SPONSORS.md`) from
[GitHub Sponsors](https://github.com/sponsors/Hkshoonya) using the
[`sponsors.yml`](workflows/sponsors.yml) workflow.

## Key concept: sponsorship is account-level

GitHub Sponsors tiers belong to the **account** (`Hkshoonya`), not to any single
repo. So a sponsor is sponsoring *you*, across **all** your projects. This system
takes that one shared list of sponsors and renders it into each project's README —
project-based **display**, account-wide **data**.

## One-time manual step: create the token (only you can do this)

The GitHub Sponsors API **cannot** be read by the built-in `GITHUB_TOKEN`, so the
workflow needs a Personal Access Token:

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**.
   Direct link: https://github.com/settings/tokens
2. **Generate new token (classic)**. Name it `sponsors-readme`.
3. Tick scopes: **`read:user`** (and **`read:org`** only if you ever set up org-level sponsorships).
4. Generate and copy the token.
5. In **each repo** that uses this workflow: **Settings → Secrets and variables → Actions → New repository secret**.
   - Name: `SPONSORS_TOKEN`
   - Value: the token you copied
   - (Tip: add it once as an **organization** secret if your repos live under an org, so you don't repeat this per repo.)

That's it — the next scheduled run (or a manual **Actions → Update Sponsors → Run workflow**) fills the walls.
If the secret is not configured yet, the workflow exits successfully without
changing the sponsor walls.

## How it refreshes

- **Daily** at 06:00 UTC (`schedule` cron), and on demand via the **Run workflow** button.
- There is **no** per-repo "sponsorship" Actions trigger — those webhooks fire at the
  account level — so a daily cron is the standard, reliable mechanism. A new sponsor
  appears within ~24h (or instantly if you hit Run workflow).

## Tiers shown

- **Featured** wall = sponsors at **$50/mo and up** (Creator / Studio / Company) — larger avatars.
- **Backers** wall = every public sponsor — smaller avatars.
- Thresholds are set with `minimum` (in **cents**) in `sponsors.yml`. Change `5000` to retune.
- Only **public** sponsors render; private sponsors are hidden (set `include-private: true` to show a redacted count).

## Reusing this in your other projects

Because the data is account-wide, every project can show the same wall. For each repo:

1. Copy `.github/workflows/sponsors.yml` into the repo.
2. Add the marker comments where you want the walls to appear:
   ```html
   <!-- featured --><!-- featured -->
   <!-- sponsors --><!-- sponsors -->
   ```
   (in `README.md`, and optionally a `SPONSORS.md`).
3. Add the `SPONSORS_TOKEN` secret (or use an org-level secret to skip this).

### DRY upgrade (optional)

To avoid copy-pasting the workflow into every repo, create a public repo named
**`Hkshoonya/.github`** with a *reusable* workflow, then each project only needs a
tiny caller:

```yaml
# .github/workflows/sponsors.yml in each project
name: Update Sponsors
on:
  schedule: [{ cron: '0 6 * * *' }]
  workflow_dispatch:
jobs:
  call:
    uses: Hkshoonya/.github/.github/workflows/sponsors.yml@main
    secrets: inherit
```

Ask me to scaffold the `Hkshoonya/.github` reusable workflow when you want this.
