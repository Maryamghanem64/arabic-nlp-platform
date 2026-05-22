<template>
  <div class="app-shell">
    <header class="topbar">
      <RouterLink to="/" class="brand" aria-label="Arabic NLP platform home">
        <span class="brand-mark">ع</span>
        <span class="brand-copy">
          <strong>Arabic NLP Lab</strong>
          <small>Comparative Research Platform</small>
        </span>
      </RouterLink>

      <nav class="nav-links" aria-label="Primary navigation">
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/compare">Compare</RouterLink>
        <RouterLink to="/analyze">Analyze</RouterLink>
        <RouterLink to="/about">About</RouterLink>
      </nav>
    </header>

    <main>
      <router-view v-slot="{ Component }">
        <transition name="page" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <footer class="footer">
      <div>
        <strong>Arabic NLP Comparative Platform v8.3</strong>
        <span>CAMeL, Farasa, Stanza, Qalsadi, AraBERT, and SinaTools evidence workspace.</span>
      </div>
      <span class="footer-tag">Research dashboard</span>
    </footer>
  </div>
</template>

<style>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 12px 30px;
  border-bottom: 1px solid rgba(220, 228, 239, 0.86);
  background: rgba(255, 255, 255, 0.86);
  backdrop-filter: blur(18px);
}

.brand {
  min-width: 260px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
  color: var(--ink);
  text-decoration: none;
}

.brand-mark {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  color: white;
  background: linear-gradient(135deg, var(--navy), var(--cyan));
  box-shadow: 0 10px 24px rgba(22, 54, 92, 0.24);
  font-family: "Noto Naskh Arabic", "Traditional Arabic", serif;
  font-size: 25px;
  font-weight: 900;
}

.brand-copy strong,
.brand-copy small {
  display: block;
  line-height: 1.15;
}

.brand-copy strong {
  font-size: 15px;
  font-weight: 900;
}

.brand-copy small {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #eef3f8;
}

.nav-links a {
  min-width: 88px;
  min-height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  color: #556173;
  font-size: 14px;
  font-weight: 850;
  text-decoration: none;
}

.nav-links a.router-link-active {
  color: var(--navy);
  background: white;
  box-shadow: 0 1px 7px rgba(23, 32, 51, 0.09);
}

main {
  flex: 1;
}

.footer {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 26px 30px;
  color: #d7dee8;
  background:
    linear-gradient(135deg, #101827, #13243a),
    radial-gradient(circle at 88% 20%, rgba(37, 99, 235, 0.26), transparent 28%);
}

.footer div {
  display: grid;
  gap: 4px;
}

.footer strong {
  font-weight: 900;
}

.footer span {
  color: #9eadbd;
  font-size: 13px;
}

.footer-tag {
  white-space: nowrap;
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

@media (max-width: 820px) {
  .topbar {
    align-items: stretch;
    flex-direction: column;
    padding: 14px;
  }

  .brand {
    min-width: 0;
  }

  .nav-links {
    overflow-x: auto;
  }

  .nav-links a {
    min-width: 78px;
  }

  .footer {
    flex-direction: column;
  }
}
</style>
