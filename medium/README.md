# Asynchronous Agentic Orchestration at Scale — GitHub Pages edition

This repository is the static article edition of “Asynchronous Agentic Orchestration at Scale”. It is deliberately built as plain HTML/CSS with local PNG assets so that GitHub Pages can serve it directly and Medium can import the published URL without depending on JavaScript, MathJax, Jekyll, or a build step.

## Publish on GitHub Pages

1. Create a new GitHub repository and upload the contents of this folder to the repository root.
2. In GitHub, open Settings → Pages.
3. Under “Build and deployment”, choose “Deploy from a branch”.
4. Select the `main` branch and `/ (root)` folder, then save.
5. Wait for GitHub to publish the site and open the Pages URL.

## Import into Medium

In Medium, use the “Import a story” option and paste the public GitHub Pages URL. The article uses semantic headings, paragraphs, blockquotes, tables, code blocks, ordinary PNG figures, and equation images. No client-side rendering is required for the article body.

After import, Medium may reflow tables and image captions according to its own editor. Review those elements once before publishing.

## Files

- `index.html` — the page Medium should import.
- `article.md` — editable source for the article.
- `style.css` — GitHub Pages styling; Medium does not depend on it.
- `assets/figures/` — article figures.
- `assets/equations/` — display equations rendered as PNG for reliable Medium import.

There are intentionally no hidden files, no `.github` directory, and no `.gitignore`.
