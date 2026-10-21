-- CH02 测试数据：FAQ、演示会话/消息、演示工单
-- 可重复执行：全部使用 INSERT IGNORE

INSERT IGNORE INTO faq (id, question, answer, category) VALUES
  (1, '退货政策是什么？', '商品签收后 7 天内，商品完好、附件齐全即可申请无理由退货；定制商品和已拆封的贴身用品除外。', '退货'),
  (2, '怎么申请换货？', '在订单详情页点击“申请换货”，选择换货原因并上传凭证，客服会在 24 小时内审核。', '换货'),
  (3, '付款后多久发货？', '现货订单在付款后 48 小时内发出；预售商品以商品页标注的发货时间为准。', '发货'),
  (4, '可以开发票吗？', '支持电子普通发票，下单时在备注中填写抬头和税号，发货后可在订单详情页下载。', '发票'),
  (5, '客服服务时间是什么时候？', '人工客服每天 9:00-21:00 在线；非服务时间可以留言，我们会在下一个服务时段回复。', '服务');

INSERT IGNORE INTO conversations (id, user_id, status) VALUES
  (9001, 'demo-seed', '已结束');

INSERT IGNORE INTO messages (id, conversation_id, role, content) VALUES
  (9001, 9001, 'user', '退货政策是什么？'),
  (9002, 9001, 'assistant', '商品签收后 7 天内，商品完好、附件齐全即可申请无理由退货。');

INSERT IGNORE INTO tickets
  (ticket_no, conversation_id, description, ticket_type, status) VALUES
  ('T20260701008', 9001, '演示工单：申请退货', '售后', '已处理');
