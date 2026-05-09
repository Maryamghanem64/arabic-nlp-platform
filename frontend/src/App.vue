<template>
  <div class="app-shell">
    <header class="topbar">
      <RouterLink to="/" class="brand" aria-label="Arabic NLP Platform home">
        <span class="brand-mark">ع</span>
        <span>
          <strong>Arabic NLP</strong>
          <small>Comparative Platform</small>
        </span>
      </RouterLink>

      <nav class="nav-links" aria-label="Primary navigation">
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/analyze">Analyze</RouterLink>
        <RouterLink to="/compare">Compare</RouterLink>
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
        <strong>Arabic NLP Comparative Platform</strong>
        <span>CAMeL Tools, Farasa, Stanza, and Qalsadi in one research interface.</span>
      </div>
      <span>v8.3</span>
    </footer>
  </div>
</template>

<style>
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #647084;
  --soft: #f5f7fb;
  --panel: #ffffff;
  --line: #dce3ee;
  --navy: #17385f;
  --blue: #2563eb;
  --teal: #0f766e;
  --violet: #7c3aed;
  --amber: #b45309;
  --green: #15803d;
  --red: #b42318;
  --shadow: 0 16px 44px rgba(23, 32, 51, 0.08);
  --radius: 8px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--soft);
  color: var(--ink);
  font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
}

button,
textarea,
input,
select {
  font: inherit;
}

a {
  color: inherit;
}

.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  height: 68px;
  padding: 0 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(14px);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  min-width: 230px;
}

.brand-mark {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: var(--navy);
  color: white;
  font-size: 22px;
  font-weight: 900;
  font-family: "Noto Naskh Arabic", "Traditional Arabic", serif;
}

.brand strong,
.brand small {
  display: block;
  line-height: 1.15;
}

.brand small {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px;
  background: #edf2f8;
  border-radius: 8px;
}

.nav-links a {
  min-width: 84px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  color: #526073;
  font-size: 14px;
  font-weight: 700;
  text-decoration: none;
}

.nav-links a.router-link-active {
  color: var(--navy);
  background: white;
  box-shadow: 0 1px 4px rgba(23, 32, 51, 0.08);
}

main {
  flex: 1;
}

.footer {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 24px 28px;
  color: #cbd5e1;
  background: #101827;
}

.footer div {
  display: grid;
  gap: 5px;
}

.footer span {
  color: #94a3b8;
  font-size: 13px;
}

.page-enter-active,
.page-leave-active {
  transition: opacity 0.16s ease, transform 0.16s ease;
}

.page-enter-from,
.page-leave-to {
  opacity: 0;
  transform: translateY(6px);
}

@media (max-width: 760px) {
  .topbar {
    height: auto;
    padding: 14px;
    align-items: stretch;
    flex-direction: column;
  }

  .brand {
    min-width: 0;
  }

  .nav-links {
    overflow-x: auto;
    justify-content: space-between;
  }

  .nav-links a {
    min-width: 76px;
  }

  .footer {
    flex-direction: column;
  }
}
</style>
