# Blog Enhancement Setup Guide

Your blog has been enhanced with a professional design inspired by the reference blog! Here's what's been added and how to configure it.

## What's New

### 1. Enhanced Post Layout
- Beautiful post navigation (Previous/Next posts)
- Social sharing buttons (Twitter, Facebook, Reddit, Email)
- Tag display for posts
- Comment section (Disqus integration)
- Social links section (GitHub, LinkedIn)
- MathJax support for mathematical equations

### 2. Professional Styling
- Modern, clean design with improved typography
- Enhanced spacing and layout
- Responsive design for mobile devices
- Hover effects and smooth transitions
- Card-style sections with subtle backgrounds

## Configuration Steps

### 1. Add Your LinkedIn Username

In `_config.yml`, I've added:
```yaml
linkedin_username: yashwanth-atla
```

Update this with your actual LinkedIn username.

### 2. Enable Comments (Optional)

To enable Giscus comments (100% free, uses GitHub Discussions):

1. Enable Discussions in your GitHub repo (Settings → Features → Discussions)
2. Go to https://giscus.app/
3. Enter your repo: `yashuatla1431/yashuatla1431.github.io`
4. Copy the configuration values
5. In `_config.yml`, uncomment and update:
   ```yaml
   giscus:
     repo: yashuatla1431/yashuatla1431.github.io
     repo_id: YOUR_REPO_ID
     category: Comments
     category_id: YOUR_CATEGORY_ID
   ```

### 3. Add Tags to Your Posts

Update your blog posts' front matter to include tags:

```yaml
---
layout: post
title: "Your Post Title"
date: 2026-01-15
categories: [category1, category2]
tags: [machine-learning, transformers, nlp]
---
```

## Files Created

1. **_includes/post_nav.html** - Previous/Next post navigation
2. **_includes/social-share.html** - Social sharing buttons
3. **_includes/social-links.html** - Your social profile links
4. **_includes/disqus.html** - Comment section integration

## Files Modified

1. **_layouts/post.html** - Enhanced with all new components
2. **assets/css/style.scss** - Added extensive styling for new features
3. **_config.yml** - Added LinkedIn username and configuration options

## Testing Locally

Run your blog locally to see the changes:

```bash
bundle exec jekyll serve
```

Then visit http://localhost:4000

## Features Breakdown

### Post Navigation
- Appears at the bottom of each post
- Shows previous and next posts with titles
- Styled with clear labels and hover effects

### Social Share Buttons
- Share to Twitter, Facebook, Reddit, or Email
- Each button has custom hover colors
- Fully responsive design

### Social Links Section
- Your GitHub and LinkedIn profiles
- Appears at the bottom of each post
- Clean, card-style design

### Tags
- Display all tags from your post
- Styled as pills/badges
- Easy to scan and read

### Comments
- Powered by Disqus
- Only appears when configured
- Integrates seamlessly with the design

## Customization

### Change Colors

Edit the CSS variables in `assets/css/style.scss`:

```scss
:root {
  --bg-color: #000000;        /* Background color */
  --text-color: #ffffff;      /* Text color */
  --text-muted: #888888;      /* Secondary text */
  --border-color: #222222;    /* Borders */
  --link-color: #ffffff;      /* Links */
  --code-bg: #111111;         /* Code blocks */
}
```

### Remove Components

If you don't want a specific component, comment it out in `_layouts/post.html`:

```html
<!-- {% include social-share.html %} -->
```

## Deployment

Push your changes to GitHub:

```bash
git add .
git commit -m "Enhance blog design with navigation, social links, and comments"
git push origin main
```

GitHub Pages will automatically rebuild your site!

## Sources

- Reference blog: https://manalelaidouni.github.io/
- Repository: https://github.com/Manalelaidouni/manalelaidouni.github.io

Enjoy your beautifully enhanced blog!
