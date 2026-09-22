/* 自定义脚本 - 由主题 footer/components/script.html 经 js.Build 自动编译加载
   不要再用 assets/js/custom.js + footer/custom.html 静态引用，那条路径会 404 */

document.addEventListener('DOMContentLoaded', function () {
  /* ========================================
     页面类型判定
     - 阅读页（content/post/ 下的文章）应保持纯粹，仅保留顶部进度条，
       关掉飘落花瓣、云朵、小树等动态装饰，避免干扰阅读
     - 独立页（About / Links / Travelling，带 .page-standalone）
       虽然也是 article-page，但保留全部装饰，和首页观感一致
     ======================================== */
  var isReading = !!document.querySelector('.article-page');
  var isStandalone = !!document.querySelector('.page-standalone');
  /* 装饰开关：独立页例外，其余非阅读页照常装饰 */
  var decorEnabled = !isReading || isStandalone;

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
  if (decorEnabled && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
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
  if (decorEnabled) {
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
     Links 页 - 把结尾的「欢迎语 + 友链格式说明」收进一张卡片
     正文模板里排在友链网格之后，裸文本飘着不好看。
     同样纯客户端重排，Markdown 原文保持可读。
     ======================================== */
  var linksContent = document.querySelector('.page-links .article-content');
  if (linksContent && linksContent.children.length) {
    var noteCard = document.createElement('section');
    noteCard.className = 'links-note';

    /* 卡片上方先插一道花枝标语卡 —— 和 About / Moments 页的段落开场一致 */
    var noteBanner = document.createElement('div');
    noteBanner.className = 'spring-banner links-note-banner';
    noteBanner.innerHTML =
      '<svg class="spring-banner-branch" viewBox="0 0 150 64" aria-hidden="true">' +
        '<path d="M6,54 C32,46 54,50 82,32 C102,20 122,22 140,28" fill="none" stroke="#c08a76" stroke-width="3" stroke-linecap="round"/>' +
        '<path d="M56,42 C64,30 70,26 78,18" fill="none" stroke="#d2a088" stroke-width="2" stroke-linecap="round"/>' +
        '<defs><g id="lkn-sakura">' +
          '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8"/>' +
          '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(72)"/>' +
          '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(144)"/>' +
          '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(216)"/>' +
          '<ellipse cx="0" cy="-4.6" rx="2.3" ry="4.2" fill="#f3b5c8" transform="rotate(288)"/>' +
          '<circle r="2.1" fill="#f8d4de"/>' +
        '</g></defs>' +
        '<use href="#lkn-sakura" x="82" y="30"/>' +
        '<use href="#lkn-sakura" x="78" y="14"/>' +
        '<use href="#lkn-sakura" x="140" y="26"/>' +
        '<circle cx="22" cy="50" r="3.6" fill="#f0a8bc"/>' +
        '<circle cx="124" cy="20" r="3" fill="#f0a8bc"/>' +
        '<circle cx="12" cy="48" r="2.4" fill="#f0a8bc"/>' +
      '</svg>' +
      '<div class="spring-banner-text">' +
        '<span class="spring-banner-title">关于这里</span>' +
        '<span class="spring-banner-sub">About this page · 一点说明</span>' +
      '</div>';

    var noteBody = document.createElement('div');
    noteBody.className = 'links-note-body';

    /* 一次性搬完所有子节点，避免边遍历边删除导致漏项 */
    while (linksContent.firstChild) {
      noteBody.appendChild(linksContent.firstChild);
    }

    /* 「关于这里」已经在标语卡里承担了标题语义，正文那个 h2 就去掉，
       否则同一个小节会出现两个一模一样的标题 */
    var dupHeading = noteBody.querySelector('h2');
    if (dupHeading && dupHeading.textContent!.indexOf('关于这里') >= 0) {
      dupHeading.parentNode!.removeChild(dupHeading);
    }

    noteCard.appendChild(noteBanner);
    noteCard.appendChild(noteBody);
    linksContent.appendChild(noteCard);
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
  if (decorEnabled) {
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
  if (decorEnabled && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
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
  if (decorEnabled) {
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
     Moments - 私密角落（描述末尾的小钥匙 + 密码解锁）

     密文由 encrypt_moments.py 生成：明文写在 private/moments.md（已 gitignore），
     只有密文 + salt + 校验值进仓库。这里做的是真解密——密码不在页面里，
     校验值是明文的 SHA-256，只用来判断「密码对不对」，反推不出密码。
     解开后的内容只留在内存 / sessionStorage，关掉标签页就忘掉。
     ======================================== */
  var secretBtn = document.querySelector('.moments-secret') as HTMLElement | null;
  var vault = document.getElementById('moments-vault');

  if (secretBtn && vault) {
    var pwInput = vault.querySelector('.moments-vault-input') as HTMLInputElement;
    var vaultBtn = vault.querySelector('.moments-vault-btn') as HTMLElement;
    var vaultMsg = vault.querySelector('.moments-vault-msg') as HTMLElement;
    var hintBtn = vault.querySelector('.moments-vault-hint-btn') as HTMLElement | null;
    var hintText = vault.querySelector('.moments-vault-hint-text') as HTMLElement | null;
    var privateBlock = document.getElementById('moments-private-block');
    var STORE_KEY = 'moments-private-open';
    var payload: any = null;

    try {
      payload = JSON.parse(vault.getAttribute('data-vault') || '{}');
    } catch (e) {
      payload = null;
    }

    function b64decode(s: string): Uint8Array {
      var bin = window.atob(s);
      var out = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    function b64encode(bytes: Uint8Array): string {
      var s = '';
      for (var i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
      return window.btoa(s);
    }

    async function deriveHmacKey(password: string, salt: Uint8Array, iterations: number) {
      var enc = new TextEncoder();
      var baseKey = await crypto.subtle.importKey(
        'raw', enc.encode(password), 'PBKDF2', false, ['deriveBits']
      );
      var bits = await crypto.subtle.deriveBits(
        { name: 'PBKDF2', salt: salt, iterations: iterations, hash: 'SHA-256' },
        baseKey,
        256
      );
      return crypto.subtle.importKey(
        'raw', bits, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']
      );
    }

    /* 与 encrypt_moments.py 完全对齐：PBKDF2 取密钥 → HMAC(key, counter) 拼流密钥 → XOR */
    async function decryptVault(password: string): Promise<string | null> {
      var salt = b64decode(payload.salt);
      var data = b64decode(payload.data);
      var hmacKey = await deriveHmacKey(password, salt, payload.iter);

      var ks = new Uint8Array(data.length);
      var off = 0;
      var blockIndex = 0;
      var ctr = new Uint8Array(4);
      while (off < data.length) {
        ctr[0] = (blockIndex >>> 24) & 255;
        ctr[1] = (blockIndex >>> 16) & 255;
        ctr[2] = (blockIndex >>> 8) & 255;
        ctr[3] = blockIndex & 255;
        var block = new Uint8Array(await crypto.subtle.sign('HMAC', hmacKey, ctr));
        for (var j = 0; j < block.length && off + j < data.length; j++) ks[off + j] = block[j];
        off += block.length;
        blockIndex++;
      }

      var plain = new Uint8Array(data.length);
      for (var k = 0; k < data.length; k++) plain[k] = data[k] ^ ks[k];

      var digest = new Uint8Array(await crypto.subtle.digest('SHA-256', plain));
      if (b64encode(digest) !== payload.check) return null;
      return new TextDecoder().decode(plain);
    }

    function setMsg(text: string, isError: boolean) {
      vaultMsg.textContent = text;
      vaultMsg.classList.toggle('is-error', !!isError);
    }

    function renderPrivate(list: Array<{ time: string; html: string }>) {
      if (!privateBlock) return;

      /* 按年份倒序分组，每组渲染成一段「年份 + 时间轴」，跟公开条目同样的分节结构。
         list 已由加密脚本按时间倒序排好，这里只要顺延切分即可。 */
      var groups: Array<{ year: string; items: Array<{ time: string; html: string }> }> = [];
      list.forEach(function (item) {
        var year = item.time.slice(0, 4);
        var last = groups[groups.length - 1];
        if (!last || last.year !== year) {
          groups.push({ year: year, items: [item] });
        } else {
          last.items.push(item);
        }
      });

      var html = '<div class="moments-private-head">这几条，只有你我知道</div>';
      groups.forEach(function (g) {
        html += '<div class="moments-year-block">';
        html += '<h2 class="moments-year">' + g.year + '</h2>';
        html += '<div class="moment-list">';
        g.items.forEach(function (item) {
          var parts = item.time.slice(0, 10).split('-');
          var label = parts[1] + ' 月 ' + parts[2] + ' 日 ' + item.time.slice(11, 16);
          html +=
            '<section class="moment-item">' +
              '<time class="moment-date" datetime="' + item.time.replace(' ', 'T') + '" title="' + item.time + '">' +
                label +
              '</time>' +
              '<div class="moment-card"><div class="moment-content">' + item.html + '</div></div>' +
            '</section>';
        });
        html += '</div></div>';
      });
      html +=
        '<div class="moments-private-foot">' +
          '<span>解开后就先放在这个标签页里，关掉它就忘了。</span>' +
          '<button type="button" class="moments-relock">重新上锁</button>' +
        '</div>';

      privateBlock.innerHTML = html;
      privateBlock.removeAttribute('hidden');

      var relock = privateBlock.querySelector('.moments-relock') as HTMLElement | null;
      if (relock) relock.addEventListener('click', lockPrivate);
    }

    function lockPrivate() {
      if (privateBlock) {
        privateBlock.innerHTML = '';
        privateBlock.setAttribute('hidden', '');
      }
      try {
        window.sessionStorage.removeItem(STORE_KEY);
      } catch (e) { /* 无痕模式下忽略 */ }

      (secretBtn as HTMLElement).classList.remove('is-unlocked');
      (secretBtn as HTMLElement).setAttribute('aria-expanded', 'false');
      vaultMsg.textContent = '';
      pwInput.value = '';
      setHintOpen(false);
    }

    async function tryUnlock() {
      var password = pwInput.value;
      if (!password) {
        setMsg('先写点什么吧', true);
        pwInput.focus();
        return;
      }
      if (!payload || !payload.data) {
        setMsg('这一页还没有上锁的内容', true);
        return;
      }
      /* WebCrypto 只在 https / localhost 下可用（局域网 http 访问会走到这里） */
      if (!window.crypto || !window.crypto.subtle) {
        setMsg('当前环境不支持加密，请改用 https 或 localhost 打开', true);
        return;
      }

      vaultBtn.setAttribute('disabled', '');
      setMsg('正在试…', false);

      try {
        var text = await decryptVault(password);
        if (text === null) {
          setMsg('好像不太对，再想想？', true);
          vault.classList.add('is-shaking');
          window.setTimeout(function () {
            vault.classList.remove('is-shaking');
          }, 460);
          pwInput.select();
        } else {
          var list = JSON.parse(text);
          try {
            window.sessionStorage.setItem(STORE_KEY, text);
          } catch (e) { /* 存不下就算了，刷新后重输 */ }

          renderPrivate(list);
          (secretBtn as HTMLElement).classList.add('is-unlocked');
          vault.setAttribute('hidden', '');
          (secretBtn as HTMLElement).setAttribute('aria-expanded', 'false');
          pwInput.value = '';
          setMsg('解开了 ' + list.length + ' 条', false);
        }
      } catch (e) {
        setMsg('出了点问题，刷新页面再试一次', true);
      }

      vaultBtn.removeAttribute('disabled');
    }

    (secretBtn as HTMLElement).addEventListener('click', function () {
      var unlocked = (secretBtn as HTMLElement).classList.contains('is-unlocked');
      /* 已解锁时再点一下 = 收起并重新上锁 */
      if (unlocked) {
        lockPrivate();
        vault.setAttribute('hidden', '');
        return;
      }

      var opened = !vault.hasAttribute('hidden');
      if (opened) {
        vault.setAttribute('hidden', '');
        (secretBtn as HTMLElement).setAttribute('aria-expanded', 'false');
      } else {
        vault.removeAttribute('hidden');
        (secretBtn as HTMLElement).setAttribute('aria-expanded', 'true');
        pwInput.focus();
      }
    });

    vaultBtn.addEventListener('click', tryUnlock);
    pwInput.addEventListener('keydown', function (e) {
      if ((e as KeyboardEvent).key === 'Enter') {
        e.preventDefault();
        tryUnlock();
      }
    });

    /* 提示：默认藏着，点一下才展开；再点一下收回去。
       按钮上的字也跟着换，免得人以为点了没反应。 */
    function setHintOpen(open: boolean) {
      if (!hintBtn || !hintText) return;

      if (open) hintText.removeAttribute('hidden');
      else hintText.setAttribute('hidden', '');

      hintBtn.setAttribute('aria-expanded', open ? 'true' : 'false');

      var label = hintBtn.querySelector('.moments-vault-hint-label');
      if (label) label.textContent = open ? '我先自己想想' : '悄悄给点提示';
    }

    if (hintBtn && hintText) {
      hintBtn.addEventListener('click', function () {
        setHintOpen(hintText.hasAttribute('hidden'));
      });
    }

    /* 同一个标签页里刷新不用重输密码 */
    (function restore() {
      var cached: string | null = null;
      try {
        cached = window.sessionStorage.getItem(STORE_KEY);
      } catch (e) {
        cached = null;
      }
      if (!cached) return;
      try {
        renderPrivate(JSON.parse(cached));
        (secretBtn as HTMLElement).classList.add('is-unlocked');
      } catch (e) {
        try {
          window.sessionStorage.removeItem(STORE_KEY);
        } catch (e2) { /* ignore */ }
      }
    })();
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
