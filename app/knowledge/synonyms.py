SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "邮费": ("运费",),
    "运费": ("邮费",),
    "到账": ("退款到账",),
    "退款到账": ("到账",),
    "保修": ("质保",),
    "质保": ("保修",),
    "续航": ("使用时间", "播放时间"),
    "使用时间": ("续航", "播放时间"),
    "播放时间": ("续航", "使用时间"),
    "发票": ("开票",),
    "开票": ("发票",),
    "换货": ("更换",),
    "更换": ("换货",),
    "退货": ("退款",),
    "退款": ("退货",),
    "降噪": ("主动降噪",),
    "主动降噪": ("降噪",),
    "配对": ("连接",),
    "连接": ("配对",),
}


def expand_synonyms(text: str) -> str:
    additions: list[str] = []
    for term, synonyms in SYNONYM_GROUPS.items():
        if term not in text:
            continue
        for synonym in synonyms:
            if (
                synonym not in text
                and synonym not in additions
                and synonym != term
            ):
                additions.append(synonym)
    if not additions:
        return text
    return f"{text} {' '.join(additions)}"
