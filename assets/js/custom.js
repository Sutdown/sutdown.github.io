document.addEventListener('DOMContentLoaded', function() {
  var toc = document.querySelector('.widget--toc #TableOfContents');
  if (!toc) return;

  var links = toc.querySelectorAll('a');
  var headings = [];

  links.forEach(function(link) {
    var href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      var id = href.substring(1);
      var heading = document.getElementById(id);
      if (heading) {
        headings.push({ link: link, heading: heading });
      }
    }
  });

  if (headings.length === 0) return;

  function highlightActive() {
    var current = null;
    headings.forEach(function(item) {
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

  window.addEventListener('scroll', function() {
    highlightActive();
  });

  highlightActive();
});
