---
layout: page
title: Categories
permalink: /categories/
---

Explore posts organized by topic:

{% assign categories = site.categories | sort %}
{% for category in categories %}
  {% assign category_name = category[0] %}
  {% assign posts = category[1] %}

  <h2 id="{{ category_name | slugify }}" style="margin-top: 2rem; color: var(--primary-color);">
    {{ category_name }}
  </h2>

  <ul class="post-list">
    {% for post in posts %}
      <li style="margin-bottom: 1.5rem;">
        <h3 style="margin-bottom: 0.25rem;">
          <a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
        </h3>
        <p style="color: var(--text-light); font-size: 0.9rem; margin-top: 0.25rem;">
          {{ post.date | date: "%B %d, %Y" }}
        </p>
        {% if post.excerpt %}
          <p style="margin-top: 0.5rem;">{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
        {% endif %}
      </li>
    {% endfor %}
  </ul>
{% endfor %}

<style>
.post-list {
  list-style: none;
  padding-left: 0;
}
</style>
