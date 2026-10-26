-- =============================================================
-- ch07 · 分层上下文 · 建表 DDL(第二步,共两步)
-- 两件事:conversations 加一列记层 1 起点;新建一张分段摘要表。
--
-- 层的边界靠用户消息 id 表达,不搬数据:
--   turn_anchor <= summary_upto_msg_id          已进摘要
--   summary_upto < turn_anchor < layer1_from    层 2,渲染成半压形态
--   turn_anchor >= layer1_from                  层 1,原样
--
-- 摘要一段一行、只追加:压完的段落不再回炉重压,一个事实只经历一次有损压缩。
-- =============================================================

SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL
    COMMENT '层1(原文)起点的用户消息 id;此 id 及之后原样,之前渲染成半压形态'
    AFTER summary_upto_msg_id;

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段,从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的第一轮用户消息 id,闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的最后一轮用户消息 id,闭区间',
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';