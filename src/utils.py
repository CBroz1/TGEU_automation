def item_has_substr(item: str, substr_list: list) -> bool:
    """Returns True if the item contains any of the substrings in the list"""
    if not isinstance(substr_list, list):
        substr_list = [substr_list]
    return any(substr.lower() in str(item).lower() for substr in substr_list)
