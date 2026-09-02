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
     春天装饰 - 樱花花瓣飘落（尊重 reduced-motion）
     ======================================== */
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var petalColors = ['#fbd6e0', '#f7c3d2', '#f3b5c8', '#f9d3da'];
    var petalSvg =
      '<svg viewBox="0 0 24 24"><path d="M12 2C7.5 7 5.5 11 5.5 14a6.5 6.5 0 0 0 13 0c0-3-2-7-6.5-12z"/></svg>';
    for (var i = 0; i < 20; i++) {
      var petal = document.createElement('div');
      petal.className = 'petal';
      var size = 14 + Math.random() * 12;
      petal.style.width = size + 'px';
      petal.style.height = size + 'px';
      petal.style.left = Math.random() * 100 + '%';
      petal.style.animationDuration = 12 + Math.random() * 14 + 's';
      petal.style.animationDelay = -Math.random() * 22 + 's';
      petal.style.setProperty('--drift', (Math.random() * 200 - 100).toFixed(0) + 'px');
      petal.style.setProperty('--petal-color', petalColors[i % petalColors.length]);
      petal.innerHTML = petalSvg;
      document.body.appendChild(petal);
    }
  }

  /* ========================================
     春天装饰 - 首页花枝横幅（仅首页）
     ======================================== */
  if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
    var list = document.querySelector('.article-list');
    if (list) {
      var banner = document.createElement('div');
      banner.className = 'spring-banner';
      banner.innerHTML =
        '<svg class="spring-banner-branch" viewBox="0 0 150 64" aria-hidden="true">' +
          '<path d="M6,54 C32,46 54,50 82,32 C102,20 122,22 140,28" fill="none" stroke="#c08a76" stroke-width="3" stroke-linecap="round"/>' +
          '<path d="M56,42 C64,30 70,26 78,18" fill="none" stroke="#d2a088" stroke-width="2" stroke-linecap="round"/>' +
          '<defs><g id="sb-sakura">' +
            '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8"/>' +
            '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(72)"/>' +
            '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(144)"/>' +
            '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(216)"/>' +
            '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(288)"/>' +
            '<circle r="2.1" fill="#f8d4de"/>' +
          '</g></defs>' +
          '<use href="#sb-sakura" x="82" y="30"/>' +
          '<use href="#sb-sakura" x="78" y="14"/>' +
          '<use href="#sb-sakura" x="140" y="26"/>' +
          '<circle cx="22" cy="50" r="3.6" fill="#f0a8bc"/>' +
          '<circle cx="124" cy="20" r="3" fill="#f0a8bc"/>' +
          '<circle cx="12" cy="48" r="2.4" fill="#f0a8bc"/>' +
        '</svg>' +
        '<div class="spring-banner-text">' +
          '<span class="spring-banner-title">欢迎来到我的小站</span>' +
          '<span class="spring-banner-sub">Spring · 樱花盛开</span>' +
        '</div>';
      list.insertAdjacentElement('beforebegin', banner);
    }
  }

  /* ========================================
     春天装饰 - 底部小树（全站）
     ======================================== */
  var tree = document.createElement('div');
  tree.className = 'spring-tree';
  tree.setAttribute('aria-hidden', 'true');
  tree.innerHTML =
    '<svg viewBox="0 0 60 92">' +
      '<ellipse cx="30" cy="88" rx="22" ry="4.5" fill="#b9d9a8"/>' +
      '<path d="M27,88 C27,60 25,52 24,40 L36,40 C35,52 33,60 33,88 Z" fill="#b98a78"/>' +
      '<circle cx="30" cy="28" r="19" fill="#c9e2b8"/>' +
      '<circle cx="30" cy="15" r="13" fill="#d9edb9"/>' +
      '<circle cx="22" cy="24" r="3" fill="#f3b5c8"/>' +
      '<circle cx="38" cy="22" r="3" fill="#f3b5c8"/>' +
      '<circle cx="30" cy="11" r="2.5" fill="#f8d4de"/>' +
      '<circle cx="33" cy="30" r="2.5" fill="#f0a8bc"/>' +
      '<circle cx="26" cy="14" r="2.2" fill="#f3b5c8"/>' +
    '</svg>';
  document.body.appendChild(tree);
});
