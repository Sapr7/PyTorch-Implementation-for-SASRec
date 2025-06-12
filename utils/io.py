def load_user_item_dict(txt_path: str) -> dict[int, list[int]]:
    user_dict = {}
    with open(txt_path, "r") as f:
        for line in f:
            u, i = map(int, line.strip().split())
            user_dict.setdefault(u, []).append(i)
    return user_dict
