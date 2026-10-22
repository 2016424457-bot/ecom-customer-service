-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL
-- 本章新建:knowledge_chunks(知识库原文权威源)、qa_extraction_staging(挖矿暂存)、
--          knowledge_mining_state(挖矿游标)
-- 向量落 Milvus 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- 可重复执行
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual / spec / qa 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  source_doc       VARCHAR(255)    NOT NULL DEFAULT ''     COMMENT '来源:知识文档文件名 / faq / conversation:<id>',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  KEY idx_source_doc (source_doc),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE IF NOT EXISTS qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  category         VARCHAR(255)    NULL                    COMMENT 'LLM 抽出的分类',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';

CREATE TABLE IF NOT EXISTS knowledge_mining_state (
  id              TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '单行游标',
  last_message_id BIGINT UNSIGNED  NOT NULL DEFAULT 0 COMMENT '已成功挖过的最大 messages.id',
  updated_at      DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话挖知识的单行游标:失败不推进,下一轮可重试';

-- =============================================================
-- 已存在旧表时补列/补索引(可重复执行)
-- 早期草稿的 knowledge_chunks 没有 source_doc,qa_extraction_staging 没有 category
-- =============================================================

SET @has_source_doc = (
  SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'knowledge_chunks'
    AND column_name = 'source_doc'
);
SET @ddl = IF(
  @has_source_doc = 0,
  'ALTER TABLE knowledge_chunks ADD COLUMN source_doc VARCHAR(255) NULL AFTER vectorize_status',
  'DO 0'
);
PREPARE migrate_stmt FROM @ddl;
EXECUTE migrate_stmt;
DEALLOCATE PREPARE migrate_stmt;

UPDATE knowledge_chunks SET source_doc = '' WHERE source_doc IS NULL
  AND EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'knowledge_chunks'
      AND column_name = 'source_doc'
  );
ALTER TABLE knowledge_chunks MODIFY COLUMN source_doc VARCHAR(255) NOT NULL DEFAULT '';

SET @has_source_index = (
  SELECT COUNT(*) FROM information_schema.statistics
  WHERE table_schema = DATABASE()
    AND table_name = 'knowledge_chunks'
    AND index_name = 'idx_source_doc'
);
SET @ddl = IF(
  @has_source_index = 0,
  'ALTER TABLE knowledge_chunks ADD INDEX idx_source_doc (source_doc)',
  'DO 0'
);
PREPARE migrate_stmt FROM @ddl;
EXECUTE migrate_stmt;
DEALLOCATE PREPARE migrate_stmt;

SET @has_staging_category = (
  SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'qa_extraction_staging'
    AND column_name = 'category'
);
SET @ddl = IF(
  @has_staging_category = 0,
  'ALTER TABLE qa_extraction_staging ADD COLUMN category VARCHAR(255) NULL AFTER source_ref',
  'DO 0'
);
PREPARE migrate_stmt FROM @ddl;
EXECUTE migrate_stmt;
DEALLOCATE PREPARE migrate_stmt;
