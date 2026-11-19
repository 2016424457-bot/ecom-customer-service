(() => {
  "use strict";

  const form = document.querySelector("[data-auth-form]");
  if (!form) return;

  const errorEl = form.querySelector(".form-error");
  const submitBtn = form.querySelector("button[type=submit]");
  const endpoint = form.dataset.endpoint || "/api/auth/login";
  const portal = form.dataset.portal || "user";
  const defaultNext = form.dataset.next || "/chat";

  function showError(message) {
    if (errorEl) errorEl.textContent = message || "";
  }

  function nextUrl() {
    const params = new URLSearchParams(window.location.search);
    const next = params.get("next");
    if (next && next.startsWith("/") && !next.startsWith("//")) {
      return next;
    }
    return defaultNext;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");

    const data = Object.fromEntries(new FormData(form));
    const username = String(data.username || "").trim();
    const password = String(data.password || "");

    if (!username || !password) {
      showError("请填写账号和密码");
      return;
    }

    if (form.dataset.confirm === "true" && password !== data.confirm) {
      showError("两次输入的密码不一致");
      return;
    }

    const payload = endpoint.includes("/register")
      ? {
          username,
          password,
          display_name: String(data.display_name || "").trim(),
        }
      : { username, password, portal };

    submitBtn.disabled = true;
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(
          typeof body.detail === "string"
            ? body.detail
            : "操作失败，请稍后重试",
        );
        return;
      }
      // A new login should not inherit the previous account's chat.
      try {
        localStorage.removeItem("ecom-chat-session-id");
        localStorage.removeItem("ecom-chat-history");
        localStorage.removeItem("ecom-chat-user-id");
      } catch {
        /* storage may be unavailable */
      }
      window.location.href = nextUrl();
    } catch {
      showError("网络异常，请稍后重试");
    } finally {
      submitBtn.disabled = false;
    }
  });
})();