<template>
  <div>
    <nav class="navbar">
      <div class="container nav-inner">
        <div class="nav-brand">
          <span class="brand-icon">🔤</span>
          <span class="brand-text">Arabic NLP Platform</span>
        </div>
        <div class="nav-links">
          <RouterLink to="/">Home</RouterLink>
          <RouterLink to="/analyze">Analyze</RouterLink>
          <RouterLink to="/compare">Compare</RouterLink>
          <RouterLink to="/about">About</RouterLink>
        </div>
      </div>
    </nav>
    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<style>
.navbar {
  background: #1F3864;
  padding: 0 0;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.nav-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-icon { font-size: 1.4rem; }

.brand-text {
  color: white;
  font-size: 1.1rem;
  font-weight: 700;
}

.nav-links {
  display: flex;
  gap: 8px;
}

.nav-links a {
  color: #BDD7EE;
  text-decoration: none;
  padding: 6px 16px;
  border-radius: 6px;
  font-size: 0.95rem;
  transition: all 0.2s;
}

.nav-links a:hover,
.nav-links a.router-link-active {
  background: rgba(255,255,255,0.15);
  color: white;
}

.main-content {
  min-height: calc(100vh - 60px);
  padding: 32px 0;
}
</style>