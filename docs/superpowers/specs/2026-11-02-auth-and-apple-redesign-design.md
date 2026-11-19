# 登录体系与 Apple 风格改版 · 设计稿

日期：2026-11-02
状态：已实现（2026-11-02）

## 意图

把现在的演示型前端改成接近真实产品的两端结构：顾客登录后只看到自己的客服对话；管理员登录后进入运营后台。同时按 Paper 水纹链接的本色重做视觉，采用 Apple 式浅色玻璃界面。

样张见 `app/static/design-preview/`（登录、管理员登录、顾客对话、审核台），作为视觉基线，确认后落到正式页面。

## 目标

- 顾客端与管理员端彻底分开：不同登录入口、不同页面、服务端拦截。
- 顾客可自助注册；管理员账号预置，不开放注册。
- 水纹背景按链接参数满屏显示（`#8f8f8f` 底、`#ffffff` 高光），不裁剪、不叠暗色蒙版。
- 全站统一 Apple 风格：浅色玻璃卡片、系统字体、深色正文、iOS 蓝主操作色、状态色点缀。
- 「数据飞轮」从顾客对话页移到管理员后台。

## 非目标

- 不做 OAuth、短信/邮箱验证、找回密码、多因素认证。
- 不做细粒度权限（只有 `user` / `admin` 两种角色）。
- 不改对话、知识库、审核、评估的业务逻辑。

## 页面与路由

顾客端：

- `GET /login` 顾客登录页；未登录访问 `/chat` 重定向到此。
- `GET /register` 顾客注册页。
- `GET /chat` 顾客对话页（需 `user` 角色）。
- `GET /` 按登录状态跳转：顾客 `/chat`，管理员 `/review-queue`，未登录 `/login`。

管理员端：

- `GET /admin/login` 管理员登录页；未登录访问后台页重定向到此。
- `GET /review-queue` 审核台（需 `admin`）。
- `GET /faith-cases` 编造台账（需 `admin`）。
- `GET /flywheel` 数据飞轮（需 `admin`，由原对话页迁出）。

认证接口：

- `POST /api/auth/register` `{username, password, display_name}` → 创建顾客账号并登录。
- `POST /api/auth/login` `{username, password, portal}`，`portal` 为 `user` / `admin`；角色不匹配时拒绝（例如顾客账号不能从管理员入口登录）。
- `POST /api/auth/logout` 清 Cookie。
- `GET /api/auth/me` 返回当前用户（供页面渲染用）。

接口保护：

- 顾客接口（`/api/chat`、`/api/actions/*`、`/api/conversations*`）要求顾客会话；`user_id` 从会话派生，不再信任请求体里的 `user_id`。
- 管理接口（`/api/review-queue*`、`/api/faith-cases*`、`/api/eval-runs*`）要求管理员会话。

## 数据与安全

- 新增 `users` 表：`id`、`username`（唯一）、`password_hash`、`role`（`user` / `admin`）、`display_name`、`created_at`。
- 用户名规则：3–32 位，字母、数字、下划线；唯一性由数据库约束保证，重复注册返回 409。
- 密码：`hashlib.scrypt`（n=2^14, r=8, p=1）+ 16 字节随机盐，存 `scrypt$n$r$p$salt$hash`。
- 会话：HMAC-SHA256 签名的 HttpOnly Cookie，内容 `{uid, role, exp}`，有效期 7 天，`SameSite=Lax`。密钥来自 `SESSION_SECRET` 环境变量，缺省时使用仅供本地开发的固定值并记录警告。
- CSRF：依赖 `SameSite=Lax`，并对所有 POST 校验 `Origin` 与请求 Host 一致；跨源请求直接 403。
- 会话用户 ID：对话与历史记录使用 `user:{id}` 作为 `user_id`，与旧的匿名 ID 隔离。
- 启动时建表并确保管理员账号存在：`ADMIN_USERNAME` / `ADMIN_PASSWORD`（开发默认 `admin` / `admin1234`），仅在账号不存在时创建，日志提示修改默认密码。
- 不新增第三方依赖（签名与哈希都用标准库）。

## 视觉规范（Apple 风格）

- 字体：`-apple-system / SF Pro / PingFang SC` 系统字体栈。
- 背景：Paper Water 着色器使用用户给定的组件参数——底图 `https://paper.design/flowers.webp`、`#8f8f8f` 底色、`#ffffff` 高光、highlights 0.4、layering 0、edges 0、waves 0、caustic 0.2、size 0.7、speed 0.1、fit cover；画布 `opacity: 1`、正常混合、满屏不裁剪，不叠加任何页面级蒙版；保留 `prefers-reduced-motion` 冻结。
- 配色取自花图：暖白玻璃、天空蓝主操作（`#1f6fa5`）、花色金点缀（`#e8890c`）、茎绿状态色（`#3e8e5a`）。
- 面板：暖白半透明玻璃（约 62–88% 白 + 18–22px 模糊）、大圆角（22–26px）、细白描边、柔和投影。
- 文字：正文深板岩蓝（`#26343c`），次要 `#5b7180`；直接压在水纹上的标题加深并带轻微白色投影。
- 状态色：绿 `#3e8e5a`、橙 `#e8890c`、红 `#d1493f`，少量使用。
- 布局：顶部薄导航栏；顾客端两栏（会话列表 + 对话，右侧值班台）；后台为页首 + 双栏/单栏内容区。
- 动效：仅用 0.18–0.2s 的淡入与按压缩放，不做过场动画。

## 前端改动

- 新增共享样式 `app/static/apple.css`（由样张 `preview.css` 收敛而来）。
- 新增页面：`login.html`、`register.html`、`admin-login.html`、`flywheel.html`。
- 重写 `chat.html`：移除「对话 / 数据飞轮」切换、移除后台入口，只保留顾客对话。
- 重写 `review-queue.html`、`faith-cases.html` 的布局与样式，保留现有交互逻辑（JS 只做必要的 DOM id 对齐）。
- 将 `chat.js` 中飞轮相关的逻辑迁到 `flywheel.js`。
- 后台页导航统一为：审核台 / 编造台账 / 数据飞轮；页面顶部显示当前管理员与「退出」。

## 迁移与兼容

- 既有 `localStorage` 的匿名 `user_id` 不再作为身份来源；登录后使用账号 ID 派生会话用户 ID。历史匿名会话不再展示（可接受）。
- 现有 API 的请求/响应结构保持不变，只增加会话校验；前端调用处补上 `credentials: 'same-origin'`（同源默认即可，仍显式保留）。
- 测试：新增 `auth_client` fixture（建用户 + 写会话 Cookie）；受影响的 5 个测试文件（聊天、抽取、审核、台账、评估接口）改用该 fixture，其余测试不变。

## 验收标准

- 未登录访问 `/chat` 跳转 `/login`；访问 `/review-queue`、`/faith-cases`、`/flywheel` 跳转 `/admin/login`。
- 顾客注册 → 登录 → 进入 `/chat`，页面无任何后台入口；管理员登录 → 进入 `/review-queue`，可访问台账与飞轮。
- 顾客账号从管理员入口登录被拒绝，反之亦然。
- 水纹背景本色满屏、无蒙版；四个页面视觉一致。
- 现有测试通过；新增认证单元测试（注册、登录、角色拒绝、Cookie 校验、页面重定向）。

## 实现说明（2026-11-02）

- 用户表在启动时通过 `User.__table__.create(checkfirst=True)` 创建，并仅在管理员账号不存在时写入种子账号；同时提供 `sql/ch10-users-ddl.sql` 供手工建库参考。
- 归属校验：聊天、恢复、待确认、反馈、工单、退款与消息列表都会校验会话属于当前用户，不属于时返回 404。
- CSRF：缺少 `Origin` 的请求放行（curl / TestClient），存在但跨源的 POST 返回 403；会话 Cookie 为 `SameSite=Lax`。
- `/api/agent` 保留给评测与开发脚本，不要求登录；其余 UI 相关接口均要求会话。
- `scripts/` 与 `evals/` 中直接调用聊天/审核接口的脚本需要先登录并携带会话 Cookie（`/api/agent` 不受影响），本次未改动这些脚本。
- 前端：顾客端与后台各自独立登录页；数据飞轮迁至 `/flywheel`；客服头像改为耳机图标，商城标志改为购物袋图标。
- 测试：`tests/test_auth.py` 覆盖密码哈希、会话签名与篡改/过期；受影响接口测试改用带用户身份的调用与会话归属数据。全量 799 项通过。