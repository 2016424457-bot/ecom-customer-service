(() => {
  "use strict";

  const path = window.location.pathname;
  const isAdminArea =
    path.startsWith("/review-queue") ||
    path.startsWith("/faith-cases") ||
    path.startsWith("/flywheel");
  const loginPath = isAdminArea ? "/admin/login" : "/login";

  const nativeFetch = window.fetch.bind(window);

  window.fetch = (input, init) => {
    return nativeFetch(input, init).then((response) => {
      if (response.status === 401) {
        const url = typeof input === "string" ? input : input.url;
        if (!url.includes("/api/auth/")) {
          window.location.href = loginPath;
        }
      }
      return response;
    });
  };

  for (const button of document.querySelectorAll("[data-logout]")) {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await nativeFetch("/api/auth/logout", { method: "POST" });
      } catch {
        /* ignore network errors while leaving */
      }
      window.location.href = loginPath;
    });
  }

  const nameEls = document.querySelectorAll("[data-user-name]");
  if (nameEls.length) {
    nativeFetch("/api/auth/me", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!data || !data.user) return;
        const name = data.user.display_name || data.user.username;
        for (const el of nameEls) el.textContent = name;
      })
      .catch(() => {});
  }
})();