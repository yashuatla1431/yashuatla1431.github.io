---
layout: page
title: Categories
permalink: /categories/
---

{% assign categories = site.categories | sort %}
{% for category in categories %}
  {% assign category_name = category[0] %}
  {% assign posts = category[1] %}

## {{ category_name }}

{% for post in posts %}
- [{{ post.title }}]({{ post.url | relative_url }}) — {{ post.date | date: "%b %d, %Y" }}
{% endfor %}

{% endfor %}
