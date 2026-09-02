/* 自定义脚本 - 由主题 footer/components/script.html 经 js.Build 自动编译加载
   不要再用 assets/js/custom.js + footer/custom.html 静态引用，那条路径会 404 */

document.addEventListener('DOMContentLoaded', function () {
  /* ========================================
     文章目录 - 滚动高亮当前章节
     ======================================== */
  var toc = document.querySelector('.widget--toc #TableOfContents');
  if (toc) {
    var links = toc.querySelectorAll('a');
    var headings: { link: HTMLAnchorElement; heading: HTMLElement }[] = [];

    links.forEach(function (link) {
      var href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        var id = href.substring(1);
        var heading = document.getElementById(id);
        if (heading) {
          headings.push({ link: link as HTMLAnchorElement, heading: heading as HTMLElement });
        }
      }
    });

    if (headings.length > 0) {
      function highlightActive() {
        var current: { link: HTMLAnchorElement; heading: HTMLElement } | null = null;
        headings.forEach(function (item) {
          var rect = item.heading.getBoundingClientRect();
          item.link.classList.remove('toc-active');
          if (rect.top <= 120) {
            current = item;
          }
        });
        if (current) {
          current.link.classList.add('toc-active');
        }
      }

      window.addEventListener('scroll', highlightActive);
      highlightActive();
    }
  }

  /* ========================================
     微交互 - 顶部阅读进度条（仅文章页）
     ======================================== */
  if (document.querySelector('.article-page')) {
    var bar = document.createElement('div');
    bar.className = 'reading-progress';
    bar.innerHTML = '<div class="reading-progress-bar"></div>';
    document.body.appendChild(bar);

    var fill = bar.firstElementChild as HTMLElement;

    function updateProgress() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var scrolled = doc.scrollTop || document.body.scrollTop;
      var pct = max > 0 ? scrolled / max : 0;
      fill.style.width = (pct * 100).toFixed(2) + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);
    updateProgress();
  }

  /* ========================================
     微交互 - 返回顶部悬浮按钮（全局）
     ======================================== */
  var btn = document.createElement('button');
  btn.className = 'back-to-top';
  btn.type = 'button';
  btn.setAttribute('aria-label', '返回顶部');
  btn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg>';
  document.body.appendChild(btn);

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      if (window.scrollY > 400) {
        btn.classList.add('is-visible');
      } else {
        btn.classList.remove('is-visible');
      }
      ticking = false;
    });
  }

  btn.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ========================================
     微交互 - 滚动进场动画（渐进增强）
     无 IntersectionObserver 或不支持时直接显示，避免内容不可见
     ======================================== */
  var targets = document.querySelectorAll(
    '.article-list article, .article-list--compact article, .right-sidebar .widget'
  );
  if (targets.length > 0) {
    if (!('IntersectionObserver' in window)) {
      targets.forEach(function (el) {
        el.classList.add('reveal', 'is-visible');
      });
    } else {
      targets.forEach(function (el) {
        el.classList.add('reveal');
      });

      var io = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add('is-visible');
              io.unobserve(entry.target);
            }
          });
        },
        { rootMargin: '0px 0px -40px 0px', threshold: 0.05 }
      );

      targets.forEach(function (el) {
        io.observe(el);
      });
    }
  }

  /* ========================================
     封面 - 客户端随机动漫图
     每次刷新每张封面从有效 API 池随机抽一个；加载失败自动换下一个，
     全部失败则回退到本地预置图（占位/兜底），避免裂图
     ======================================== */
  var coverApis = [
    'https://api.anosu.top/img/?sort=pc&num=1/',
    'https://api.anosu.top/img/?sort=pixiv&num=1/',
    'https://t.alcy.cc/ycy',
    'https://t.alcy.cc/pc',
    'https://t.alcy.cc/moe',
    'https://t.alcy.cc/fj',
    'https://t.alcy.cc/tx',
    'https://imgapi.xl0408.top/index.php',
    'https://www.dmoe.cc/random.php',
    'https://img.paulzzh.com/touhou/random',
    'https://api.mtyqx.cn/tapi/random.php'
  ];

  document.querySelectorAll<HTMLImageElement>('.article-image--cover img').forEach(function (img) {
    var fallback = img.getAttribute('src') || '';
    var key = img.getAttribute('data-key') || location.pathname;
    var tried: string[] = [];
    var isFromApi = false;

    function cacheRealUrl() {
      // 缓存重定向后的真实图床 URL，保证列表页与详情页同图
      var real = img.currentSrc || img.src;
      if (isFromApi && real && real.indexOf('http') === 0) {
        try {
          sessionStorage.setItem('cover:' + key, real);
        } catch (e) {
          /* ignore */
        }
      }
    }

    function pickNext() {
      var pool = coverApis.filter(function (a) {
        return tried.indexOf(a) === -1;
      });
      if (pool.length === 0) {
        img.onerror = null;
        isFromApi = false;
        if (fallback) img.src = fallback;
        return;
      }
      var next = pool[Math.floor(Math.random() * pool.length)];
      tried.push(next);
      isFromApi = true;
      img.src = next;
    }

    img.onerror = function () {
      pickNext();
    };
    img.onload = function () {
      cacheRealUrl();
    };

    var cached: string | null = null;
    try {
      cached = sessionStorage.getItem('cover:' + key);
    } catch (e) {
      /* ignore */
    }

    if (cached) {
      isFromApi = true;
      img.src = cached;
    } else {
      pickNext();
    }
  });
});
