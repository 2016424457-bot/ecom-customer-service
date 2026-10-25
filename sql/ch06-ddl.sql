-- =============================================================
-- ch06 · 意图路由与退款子流程 · 建表 DDL
-- 本章新建:refund_requests(退款单)
-- 用户在前端退款表单点提交后落库,单号 RYYYYMMDDNNN,同一天从 001 起递增
-- 换货/换新不进退款单:判定可处理后走售后工单(ticket_type=售后),request_type 只保留退款/退货
-- 可重复执行
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS refund_requests (
  refund_no       VARCHAR(32)      NOT NULL                COMMENT '退款单号:RYYYYMMDDNNN,当天从 001 起递增',
  conversation_id BIGINT UNSIGNED  NOT NULL                COMMENT '来源会话',
  order_id        VARCHAR(32)      NOT NULL                COMMENT '订单号',
  request_type    ENUM('退款','退货') NOT NULL             COMMENT '申请类型:换货/换新走售后工单,不进这张表',
  reason_code     VARCHAR(32)      NOT NULL                COMMENT '固定类目 code:quality / wrong_item / damaged / not_wanted / spec_mismatch / other',
  reason_label    VARCHAR(64)      NOT NULL                COMMENT '固定类目文案,提交时按 code 快照下来',
  note            TEXT             NULL                    COMMENT '用户备注,可为空',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '提交时间',
  PRIMARY KEY (refund_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_refund_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch06 退款单:用户提交的退款/退货申请';
