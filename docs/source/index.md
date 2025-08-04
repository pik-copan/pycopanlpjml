---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for copan:LPJmL, with links to the rest of the site.
html_theme.sidebar_secondary.remove: true
---

## copan:LPJmL documentation

<br>

```{image} _static/logo.svg
:alt: copan:LPJmL Logo
:class: dark-light p-2
:width: 500px
:align: center
:target: https://github.com/pik-copan/pycopanlpjml
```

<br>
<br>

copan:LPJmL (library `pycopanlpjml`) is an open World-Earth modeling framework extending copan:CORE, integrating LPJmL as the Earth system interface for comprehensive social-ecological simulations.

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} <span class="small-heading">📖 User Guide</span>
:link: user-guide/index
:link-type: doc

*New here or need more info?*  
Start with the installation and overview, then dive into the user guide for
details on the framework and its components.

+++  
[Learn more »](user-guide/index)
:::

:::{grid-item-card} <span class="small-heading">🔍 API reference</span>
:link: api/index
:link-type: doc

*Looking for function or class details?*  
Check the API reference for complete documentation of all public interfaces.
+++  
[Learn more »](api/index)
:::

:::{grid-item-card} <span class="small-heading">🗿 Examples</span>
:link: https://github.com/pik-copan/pycopanlpjml-examples
:link-type: url

*Want to see copan:LPJmL in action?*  
Check out the examples to see how the framework can be used in practice.

+++  
[Learn more »](examples/index)
:::

::::

---

<h2 class="small-heading">🌐 Resources</h2>

- copan:LPJmL [Homepage](https://copanlpjml.pik-potsdam.de)
- copan:LPJmL [Documentation](https://copanlpjml.pik-potsdam.de/docs)
- copan:LPJmL [Source code](https://github.com/pik-copan/pycopanlpjml)

```{toctree}
:hidden:
:maxdepth: 1
user-guide/index
api/index
examples/index
```
<h2 class="small-heading">📰 News</h2>

News from our [copan:LPJmL Blog](blog/index). See also the
[Changelog](about/changelog).

```{postlist} 3
:format: "{title}"
:tags: announcement
:excerpts:
:expand: Read more ...
```



<h2 class="small-heading">ℹ️ About</h2>

More information on licensing, contributors, and development.

```{toctree}
:maxdepth: 2
about/index
```
