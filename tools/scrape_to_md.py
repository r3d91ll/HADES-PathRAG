#!/usr/bin/env python3
"""scrape_to_md.py  — *rev-4*
Standalone CLI scraper that converts web pages (or whole sites) to Markdown via **ScrapeGraph-AI**.

Changes in rev-4
----------------
* **Add `--model-tokens` flag** to properly set context length for models
* **Fix duplicate `headless` bug** — passed as top-level flag only
* Add `--raw-model` switch: send the model name *exactly* as typed (skip provider prefix logic)
* Clarify vLLM aliasing: if your server exposes `Qwen/Qwen3-14B`, either run
  `vllm serve Qwen/Qwen3-14B --served-model-name Qwen3-14B` *or* call this script with
  `--model Qwen/Qwen3-14B --raw-model`
* Add duplicate file skipping to avoid processing already scraped pages
* Clean up imports and fix lint issues
"""

from __future__ import annotations

import argparse, os, sys, re, time, queue, urllib.parse as ul
from pathlib import Path
from typing import Any, List, Dict, Optional

import requests
import urllib.parse
from bs4 import BeautifulSoup
from scrapegraphai.graphs import SmartScraperGraph
try:
    from markdownify import markdownify as html2md
except ImportError:
    html2md = lambda x: re.sub(r"<[^>]+>", "", x)

# ────────────────────────── helpers ────────────────────────── #


def slugify(url: str) -> str:
    """Make a filesystem-safe slug that contains both domain *and* path.

    Examples
    --------
    https://docs.example.com           → docs-example-com
    https://docs.example.com/guide/cli → docs-example-com-guide-cli
    """
    # Strip protocol
    slug = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)

    # Remove trailing slash and URL fragments/query
    slug = urllib.parse.urlparse(slug)._replace(query="", fragment="").geturl().rstrip("/")

    # Replace any non-alphanum chars (including '/' from path) with single dashes
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug)

    # Collapse multiple dashes and trim length
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:120]

    return slug.lower()


def normalize(url: str) -> str:
    p = ul.urlparse(url); url = p._replace(fragment="").geturl(); return url[:-1] if url.endswith('/') else url

def same_domain(link: str, dom: str) -> bool:
    return ul.urlparse(link).netloc == dom

def robots_ok(url: str, ua: str = "*") -> bool:
    from urllib.robotparser import RobotFileParser
    p = ul.urlparse(url); rb = RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try: rb.read(); return rb.can_fetch(ua, url)
    except Exception: return True

def extract_links(url: str, html: str, same_domain_only: bool = True) -> List[str]:
    dom = ul.urlparse(url).netloc; soup = BeautifulSoup(html, "html.parser"); out=[]
    for tag in soup.find_all("a", href=True):
        href = tag['href']
        if not href or href.startswith(("#","javascript:","mailto:")): continue
        if not href.startswith("http"): href = ul.urljoin(url, href)
        href = normalize(href)
        if not same_domain_only or same_domain(href, dom): out.append(href)
    return out

# ────────────────────────── ScrapeGraph glue ────────────────────────── #

def build_graph(prompt: str, url: str, model: str, api_key: str, base_url: str, headless: bool,
                verbose: bool, raw_model: bool=False, model_tokens: Optional[int]=None) -> SmartScraperGraph:
    if base_url and "/" not in model and not raw_model:
        model = f"openai/{model}"
    if verbose:
        print(f"Using model   → {model}\nEndpoint       → {base_url or 'api.openai.com'}")
    cfg: Dict[str, Any] = {
        "llm": {
            "model": model,
            "api_key": api_key or "dummy",
            "base_url": base_url,
            "temperature": 0.2,
        },
        "headless": headless,
        "verbose": verbose,
    }
    if model_tokens:
        cfg["llm"]["model_tokens"] = model_tokens
    return SmartScraperGraph(prompt=prompt, source=url, config=cfg)

def dict_to_md(obj: Any, lvl:int=1) -> str:
    lines: List[str]=[]
    if isinstance(obj, dict):
        for k,v in obj.items(): lines.append(f"{'#'*lvl} {k}\n{dict_to_md(v,lvl+1)}")
    elif isinstance(obj, list):
        for it in obj: lines.append(dict_to_md(it,lvl))
    else:
        lines.append(str(html2md(obj)))
    return "\n".join(lines)

def scrape_one(url: str, prompt: str, model: str, api_key: str, base_url: str, outdir: Path,
               headless: bool, verbose: bool, raw_model: bool, model_tokens: Optional[int]=None,
               retries:int=3, delay:int=5)->Optional[Path]:
    for a in range(1,retries+1):
        try:
            g = build_graph(prompt,url,model,api_key,base_url,headless,verbose,raw_model,model_tokens)
            md = dict_to_md(g.run())
            # build destination filename
            path = outdir / f"{slugify(url)}.md"

            # ───── duplicate-protection ────
            if path.exists():                             # ← NEW early-exit
                if verbose:
                    print(f"[»] {path.name} already exists – skipping")
                return path                               # keep original file
            # ───────────────────────────────

            # write fresh file
            path.write_text(
                f"---\nsource: {url}\nscraped_at: {time.strftime('%FT%T')}\n---\n\n" + md,
                encoding="utf-8",
            )

            if verbose: print(f"[✓] {url} → {path}"); return path
        except Exception as e:
            if verbose: print(f"[!] {url} failed ({a}/{retries}): {e}")
            if a<retries: time.sleep(delay)
    return None

# ────────────────────────── crawling ────────────────────────── #

def crawl(seed:str, depth:int, max_pages:int, same_dom:bool, respect_rb:bool, verb:bool)->List[str]:
    seed=normalize(seed); dom=ul.urlparse(seed).netloc; q=queue.Queue(); q.put((seed,0)); seen:set[str]=set(); out:List[str]=[]
    while not q.empty() and len(out)<max_pages:
        url,lvl=q.get();
        if url in seen: continue; seen.add(url)
        if respect_rb and not robots_ok(url):
            if verb: print(f"[robots] {url} blocked"); continue
        try:
            r=requests.get(url,timeout=10); r.raise_for_status(); out.append(url)
            if verb: print(f"[crawl] {url} ✓ depth {lvl}")
            if lvl<depth:
                for link in extract_links(url,r.text,same_dom):
                    if link not in seen: q.put((link,lvl+1))
        except Exception as e:
            if verb: print(f"[crawl] {url} ✗ {e}")
    return out

# ────────────────────────── CLI ────────────────────────── #

def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description="Scrape pages → MD via ScrapeGraph-AI")
    p.add_argument('urls',nargs='*')
    p.add_argument('-f','--file',type=Path)
    p.add_argument('-o','--outdir',type=Path,default=Path.cwd()/"markdown_output")
    p.add_argument('-p','--prompt',default='Extract the title and main body text.')
    p.add_argument('-m','--model',default='gpt-4o-mini')
    p.add_argument('--base-url',default='')
    p.add_argument('-k','--api-key',default=os.getenv('OPENAI_API_KEY',''))
    p.add_argument('--raw-model',action='store_true',help='Send model string exactly (no auto openai/ prefix)')
    p.add_argument('--model-tokens',type=int,help='Context length of the model (sets llm.model_tokens)')
    p.add_argument('--no-headless',dest='headless',action='store_false',default=True)
    p.add_argument('-v','--verbose',action='store_true')
    p.add_argument('--crawl',action='store_true')
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--max-pages',type=int,default=100)
    p.add_argument('--same-domain-only',action='store_true',default=True)
    p.add_argument('--ignore-robots',dest='respect_robots',action='store_false')
    return p.parse_args()


def main()->None:
    a=parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    seeds:List[str]=a.urls
    if a.file: seeds.extend([l.strip() for l in a.file.read_text().splitlines() if l.strip()])
    if not seeds: sys.exit('No URLs supplied.')
    if a.verbose: print(f"Outdir → {a.outdir}\nCrawl → {a.crawl}")
    if a.crawl:
        for seed in seeds:
            d=a.outdir/slugify(ul.urlparse(seed).netloc); d.mkdir(exist_ok=True)
            targets=crawl(seed,a.depth,a.max_pages,a.same_domain_only,a.respect_robots,a.verbose)
            ok=sum(1 for u in targets if scrape_one(u,a.prompt,a.model,a.api_key,a.base_url,d,
                                                   a.headless,a.verbose,a.raw_model,a.model_tokens))
            print(f"[site] {seed} → {ok}/{len(targets)} pages scraped")
    else:
        ok=sum(1 for u in seeds if scrape_one(u,a.prompt,a.model,a.api_key,a.base_url,a.outdir,
                                      a.headless,a.verbose,a.raw_model,a.model_tokens))
        print(f"Done. {ok}/{len(seeds)} files → {a.outdir}")

if __name__=='__main__':
    main()
