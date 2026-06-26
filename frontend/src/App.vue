<template>
  <div id="app" class="platform-shell">
    <header class="platform-header">
      <div class="header-inner">
        <RouterLink to="/" class="header-brand" aria-label="Comparative Arabic NLP Platform home">
          <span class="brand-icon">A</span>
          <div>
            <div class="brand-title">Comparative Arabic NLP Platform</div>
            <div class="brand-sub">Unified analysis, comparison, fusion, and evaluation</div>
          </div>
        </RouterLink>

        <nav class="header-nav" aria-label="Primary navigation">
          <RouterLink to="/" class="nav-item">Dashboard</RouterLink>
          <RouterLink to="/analyze" class="nav-item">Analyze</RouterLink>
          <RouterLink to="/smart" class="nav-item nav-item--featured">Fusion</RouterLink>
          <RouterLink to="/compare" class="nav-item">Compare</RouterLink>
          <RouterLink to="/evaluate" class="nav-item">Evaluation</RouterLink>
          <RouterLink to="/reports" class="nav-item">Reports</RouterLink>
        </nav>
      </div>
    </header>

    <main class="platform-main">
      <router-view v-slot="{ Component }">
        <transition name="page" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <footer class="platform-footer">
      <div>
        <strong>Comparative Arabic NLP Platform</strong>
        <span>CAMeL, Farasa, Stanza, Qalsadi, UDPipe, and AlKhalil workbench.</span>
      </div>
      <span class="footer-tag">Research dashboard</span>
    </footer>
  </div>
</template>

<style>
.platform-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.platform-header {
  position: sticky;
  top: 0;
  z-index: 100;
  border-bottom: 1px solid rgba(226, 232, 240, 0.88);
  background: rgba(7, 15, 30, 0.72);
  backdrop-filter: blur(18px);
}

.header-inner {
  width: min(1360px, calc(100% - 32px));
  min-height: 76px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}

.header-brand {
  min-width: 260px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  color: #fff;
  text-decoration: none;
}

.brand-icon {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  color: white;
  background: linear-gradient(135deg, var(--cyan), var(--c-accent));
  box-shadow: 0 12px 28px rgba(22, 54, 92, 0.28);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.brand-title,
.brand-sub {
  display: block;
  line-height: 1.15;
}

.brand-title {
  font-size: 15px;
  font-weight: 600;
}

.brand-sub {
  margin-top: 3px;
  color: rgba(255, 255, 255, 0.64);
  font-size: 12px;
}

.header-nav {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 12px 30px rgba(5, 15, 30, 0.18);
}

.nav-item {
  min-width: 88px;
  min-height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  color: rgba(255, 255, 255, 0.76);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
}

.nav-item:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.nav-item.router-link-active {
  color: #fff;
  background: rgba(255, 255, 255, 0.14);
}

.nav-item--featured {
  color: #111827;
  background: linear-gradient(135deg, #E0E7FF, #D1FAE5);
  border: 1px solid rgba(255, 255, 255, 0.18);
  font-weight: 600;
}

.nav-item--featured.router-link-active {
  color: #111827;
  background: linear-gradient(135deg, #E0E7FF, #D1FAE5);
}

.platform-main {
  flex: 1;
  width: min(1360px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 68px;
}

.platform-footer {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 26px 30px;
  color: rgba(255, 255, 255, 0.88);
  background:
    linear-gradient(135deg, #08111f, #0f172a 55%, #111f38),
    radial-gradient(circle at 88% 20%, rgba(79, 70, 229, 0.26), transparent 28%);
}

.platform-footer div {
  display: grid;
  gap: 4px;
}

.platform-footer strong {
  font-weight: 600;
}

.platform-footer span {
  color: rgba(255, 255, 255, 0.68);
  font-size: 13px;
}

.footer-tag {
  white-space: nowrap;
  color: rgba(255, 255, 255, 0.86);
}

.page-enter-active,
.page-leave-active {
  transition: opacity 0.16s ease, transform 0.16s ease;
}

.page-enter-from,
.page-leave-to {
  opacity: 0;
  transform: translateY(8px);
}

@media (max-width: 980px) {
  .header-inner {
    align-items: stretch;
    flex-direction: column;
    padding: 14px 0;
  }

  .header-brand {
    min-width: 0;
  }

  .header-nav {
    overflow-x: auto;
  }

  .nav-item {
    min-width: 80px;
  }
}

@media (max-width: 820px) {
  .platform-main {
    width: min(100% - 24px, 1360px);
    padding-top: 22px;
  }

  .platform-footer {
    flex-direction: column;
  }
}
</style>
