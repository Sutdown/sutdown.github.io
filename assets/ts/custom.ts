/* 自定义脚本 - 由主题 footer/components/script.html 经 js.Build 自动编译加载
   不要再用 assets/js/custom.js + footer/custom.html 静态引用，那条路径会 404 */

document.addEventListener('DOMContentLoaded', function () {
  /* ========================================
     页面类型判定
     - 阅读页（文章详情 / 独立页）应保持纯粹，仅保留顶部进度条，
       关掉飘落花瓣、云朵、轨迹、光斑、小树等动态装饰，避免干扰阅读
     ======================================== */
  var isReading = !!document.querySelector('.article-page');
  var isHome = window.location.pathname === '/' || window.location.pathname === '/index.html';

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
     微交互 - 返回顶部进度环（全局）
     圆环随滚动进度填充，复用 rAF 节流
     ======================================== */
  var btn = document.createElement('button');
  btn.className = 'back-to-top';
  btn.type = 'button';
  btn.setAttribute('aria-label', '返回顶部');
  btn.innerHTML =
    '<svg class="progress-ring" viewBox="0 0 44 44" aria-hidden="true">' +
      '<circle class="ring-bg" cx="22" cy="22" r="19"></circle>' +
      '<circle class="ring-fg" cx="22" cy="22" r="19"></circle>' +
    '</svg>' +
    '<svg class="back-to-top-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="18 15 12 9 6 15"></polyline></svg>';
  document.body.appendChild(btn);

  var ringFg = btn.querySelector('.ring-fg') as SVGCircleElement;
  var RING_CIRC = 2 * Math.PI * 19; // r=19，约 119.38

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var scrolled = doc.scrollTop || document.body.scrollTop;
      var progress = max > 0 ? scrolled / max : 0;
      if (ringFg) {
        ringFg.style.strokeDashoffset = (RING_CIRC * (1 - progress)).toFixed(2);
      }
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
  window.addEventListener('resize', onScroll);
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
     春天装饰 - 樱花花瓣飘落（尊重 reduced-motion；阅读页关闭）
     ======================================== */
  if (!isReading && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
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
     春天装饰 - 底部小树（非阅读页）
     ======================================== */
  if (!isReading) {
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
  }

  /* ========================================
     春天装饰 - 鼠标樱花轨迹（尊重 reduced-motion；非阅读页）
     ======================================== */
  if (!isReading && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var trailSvg =
      '<svg viewBox="0 0 24 24"><path d="M12 2C7.5 7 5.5 11 5.5 14a6.5 6.5 0 0 0 13 0c0-3-2-7-6.5-12z"/></svg>';
    var trailColors = ['#fbd6e0', '#f7c3d2', '#f3b5c8', '#f9d3da'];
    var lastTrail = 0;
    document.addEventListener('mousemove', function (e) {
      var now = Date.now();
      if (now - lastTrail < 90) return;
      lastTrail = now;
      var el = document.createElement('div');
      el.className = 'petal-trail';
      var size = 8 + Math.random() * 8;
      el.style.width = size + 'px';
      el.style.height = size + 'px';
      el.style.left = e.clientX + 'px';
      el.style.top = e.clientY + 'px';
      el.style.setProperty('--petal-color', trailColors[Math.floor(Math.random() * trailColors.length)]);
      el.style.setProperty('--tx', (Math.random() * 60 - 30).toFixed(0) + 'px');
      el.style.setProperty('--ty', (Math.random() * 50 - 10).toFixed(0) + 'px');
      el.innerHTML = trailSvg;
      document.body.appendChild(el);
      window.setTimeout(function () {
        el.remove();
      }, 1200);
    });
  }

  /* ========================================
     精致装饰 - 代码块语言标签
     ======================================== */
  var highlights = document.querySelectorAll('.article-content .highlight');
  highlights.forEach(function (el) {
    var code = el.querySelector('code[data-lang]');
    if (code) {
      var lang = code.getAttribute('data-lang') || '';
      if (lang) {
        var label = document.createElement('span');
        label.className = 'code-lang-label';
        label.textContent = lang;
        el.appendChild(label);
      }
    }
  });

  /* ========================================
     实用功能 - 代码块复制按钮
     ======================================== */
  var copySvg =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  var checkSvg =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

  function fallbackCopy(text: string) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
    } catch (e) {}
    document.body.removeChild(ta);
  }

  highlights.forEach(function (el) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy-btn';
    btn.setAttribute('aria-label', '复制代码');
    btn.innerHTML = copySvg;
    btn.addEventListener('click', function () {
      var codes = el.querySelectorAll('code');
      var codeEl = el.querySelector('code[data-lang]') || (codes.length ? codes[codes.length - 1] : null);
      if (!codeEl) return;
      var text = codeEl.innerText || '';
      var done = function () {
        btn.innerHTML = checkSvg;
        btn.classList.add('is-copied');
        window.setTimeout(function () {
          btn.innerHTML = copySvg;
          btn.classList.remove('is-copied');
        }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {
          fallbackCopy(text);
          done();
        });
      } else {
        fallbackCopy(text);
        done();
      }
    });
    el.appendChild(btn);
  });

  /* ========================================
     实用功能 - 正文插图加载渐显（降级安全）
     ======================================== */
  var contentImgs = document.querySelectorAll('.article-content img');
  contentImgs.forEach(function (img) {
    if (img.complete && img.naturalWidth > 0) return;
    img.classList.add('img-fade');
    img.addEventListener('load', function () {
      img.classList.remove('img-fade');
    });
    img.addEventListener('error', function () {
      img.classList.remove('img-fade');
    });
  });

  /* ========================================
     精致装饰 - 卡片鼠标跟随光斑（非阅读页）
     ======================================== */
  if (!isReading) {
    var spotCards = document.querySelectorAll(
      '.widget, .article-list article, .article-list--compact article'
    );
    spotCards.forEach(function (card) {
      card.addEventListener('mousemove', function (e) {
        var rect = card.getBoundingClientRect();
        card.style.setProperty('--mx', (e.clientX - rect.left).toFixed(0) + 'px');
        card.style.setProperty('--my', (e.clientY - rect.top).toFixed(0) + 'px');
      });
    });
  }

  /* ========================================
     春天装饰 - 云朵缓慢飘过（非阅读页）
     ======================================== */
  if (!isReading && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var cloudSvg =
      '<svg viewBox="0 0 120 50"><path d="M30,40 C14,40 8,32 14,26 C16,16 28,10 38,16 C46,8 60,8 66,16 C80,12 92,20 88,30 C96,32 96,42 84,42 Z" fill="#ffffff" opacity="0.6"/></svg>';
    for (var ci = 0; ci < 3; ci++) {
      var cloud = document.createElement('div');
      cloud.className = 'spring-cloud';
      cloud.style.top = 6 + ci * 13 + '%';
      cloud.style.width = 110 + ci * 45 + 'px';
      cloud.style.animationDuration = 80 + ci * 30 + 's';
      cloud.style.animationDelay = -ci * 35 + 's';
      cloud.innerHTML = cloudSvg;
      document.body.appendChild(cloud);
    }
  }

  /* ========================================
     春天装饰 - 导航栏下静止花枝（非阅读页）
     ======================================== */
  if (!isReading) {
    var branch = document.createElement('div');
    branch.className = 'nav-branch';
    branch.setAttribute('aria-hidden', 'true');
    branch.innerHTML =
      '<svg viewBox="0 0 400 70" preserveAspectRatio="xMidYMin slice">' +
        '<path d="M0,36 C60,28 120,40 180,30 C240,22 320,36 400,26" fill="none" stroke="#c9a992" stroke-width="2.5" stroke-linecap="round"/>' +
        '<path d="M150,32 C158,24 168,22 176,18" fill="none" stroke="#c9a992" stroke-width="2" stroke-linecap="round"/>' +
        '<g fill="#f3b5c8" stroke="#c9a992" stroke-width="1">' +
          '<ellipse cx="180" cy="26" rx="2.5" ry="4.5"/>' +
          '<ellipse cx="180" cy="26" rx="2.5" ry="4.5" transform="rotate(72,180,30)"/>' +
          '<ellipse cx="180" cy="26" rx="2.5" ry="4.5" transform="rotate(144,180,30)"/>' +
          '<ellipse cx="180" cy="26" rx="2.5" ry="4.5" transform="rotate(216,180,30)"/>' +
          '<ellipse cx="180" cy="26" rx="2.5" ry="4.5" transform="rotate(288,180,30)"/>' +
        '</g>' +
        '<circle cx="180" cy="30" r="2" fill="#f8d4de"/>' +
        '<circle cx="90" cy="34" r="3" fill="#f3b5c8" stroke="#c9a992" stroke-width="1"/>' +
        '<circle cx="290" cy="28" r="3" fill="#f3b5c8" stroke="#c9a992" stroke-width="1"/>' +
      '</svg>';
    document.body.appendChild(branch);
  }

  /* ========================================
     交互组件 - 图片灯箱（Lightbox）
     点击正文图片全屏查看，支持键盘切换
     ======================================== */
  var lightboxImgs = document.querySelectorAll('.article-content img');
  if (lightboxImgs.length > 0) {
    var lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', '图片查看器');
    lightbox.innerHTML =
      '<div class="lightbox-backdrop"></div>' +
      '<button class="lightbox-btn lightbox-close" type="button" aria-label="关闭">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>' +
      '</button>' +
      '<button class="lightbox-btn lightbox-prev" type="button" aria-label="上一张">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>' +
      '</button>' +
      '<button class="lightbox-btn lightbox-next" type="button" aria-label="下一张">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>' +
      '</button>' +
      '<figure class="lightbox-figure">' +
        '<img class="lightbox-img" alt="">' +
        '<figcaption class="lightbox-caption"></figcaption>' +
        '<span class="lightbox-counter"></span>' +
      '</figure>';
    document.body.appendChild(lightbox);

    var lightboxImg = lightbox.querySelector('.lightbox-img') as HTMLImageElement;
    var lightboxCaption = lightbox.querySelector('.lightbox-caption') as HTMLElement;
    var lightboxCounter = lightbox.querySelector('.lightbox-counter') as HTMLElement;
    var currentIndex = 0;

    function showLightbox(index: number) {
      currentIndex = index;
      var target = lightboxImgs[index];
      var src = target.getAttribute('src') || '';
      var alt = target.getAttribute('alt') || '';
      lightboxImg.setAttribute('src', src);
      lightboxImg.setAttribute('alt', alt);
      // alt 为空或纯数字（如 "1"）时不显示 caption
      if (alt && !/^\d+$/.test(alt)) {
        lightboxCaption.textContent = alt;
        lightboxCaption.style.display = '';
      } else {
        lightboxCaption.textContent = '';
        lightboxCaption.style.display = 'none';
      }
      lightboxCounter.textContent = (index + 1) + ' / ' + lightboxImgs.length;
      lightbox.classList.add('is-open');
      document.body.classList.add('lightbox-open');
    }
    function hideLightbox() {
      lightbox.classList.remove('is-open');
      document.body.classList.remove('lightbox-open');
      window.setTimeout(function () {
        lightboxImg.removeAttribute('src');
      }, 300);
    }
    function nextImage() {
      showLightbox((currentIndex + 1) % lightboxImgs.length);
    }
    function prevImage() {
      showLightbox((currentIndex - 1 + lightboxImgs.length) % lightboxImgs.length);
    }

    lightboxImgs.forEach(function (img, i) {
      img.classList.add('lightbox-target');
      img.addEventListener('click', function (e) {
        e.preventDefault();
        showLightbox(i);
      });
    });

    (lightbox.querySelector('.lightbox-close') as HTMLElement).addEventListener('click', hideLightbox);
    (lightbox.querySelector('.lightbox-backdrop') as HTMLElement).addEventListener('click', hideLightbox);
    (lightbox.querySelector('.lightbox-prev') as HTMLElement).addEventListener('click', prevImage);
    (lightbox.querySelector('.lightbox-next') as HTMLElement).addEventListener('click', nextImage);

    document.addEventListener('keydown', function (e) {
      if (!lightbox.classList.contains('is-open')) return;
      if (e.key === 'Escape') {
        hideLightbox();
      } else if (e.key === 'ArrowRight') {
        nextImage();
      } else if (e.key === 'ArrowLeft') {
        prevImage();
      }
    });
  }

  /* ========================================
     交互组件 - 移动端底部导航
     首页 / 归档 / 搜索 / 返回顶部
     ======================================== */
  var mobileNav = document.createElement('nav');
  mobileNav.className = 'mobile-nav';
  mobileNav.setAttribute('aria-label', '移动端导航');
  var currentPath = window.location.pathname;

  var navItems = [
    {
      href: '/',
      label: '首页',
      icon: '<path d="M3 10.5L12 3l9 7.5"></path><path d="M5 9.5V21h14V9.5"></path>'
    },
    {
      href: '/archives/',
      label: '归档',
      icon: '<circle cx="12" cy="12" r="9"></circle><polyline points="12 7 12 12 15 14"></polyline>'
    },
    {
      href: '/search/',
      label: '搜索',
      icon: '<circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.5" y2="16.5"></line>'
    }
  ];

  var navHtml = '';
  navItems.forEach(function (item) {
    var isActive =
      currentPath === item.href ||
      (item.href !== '/' && currentPath.indexOf(item.href) === 0);
    navHtml +=
      '<a class="mobile-nav-item' + (isActive ? ' is-active' : '') + '" href="' + item.href + '"' +
        (isActive ? ' aria-current="page"' : '') + '>' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + item.icon + '</svg>' +
        '<span>' + item.label + '</span>' +
      '</a>';
  });
  navHtml +=
    '<button class="mobile-nav-item mobile-nav-top" type="button" aria-label="返回顶部">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="18 15 12 9 6 15"></polyline></svg>' +
      '<span>顶部</span>' +
    '</button>';
  mobileNav.innerHTML = navHtml;
  document.body.appendChild(mobileNav);

  (mobileNav.querySelector('.mobile-nav-top') as HTMLElement).addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
