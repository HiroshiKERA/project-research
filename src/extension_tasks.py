"""発展課題で比較する足し算の表記。"""


def format_problem(problem, answer):
    """両入力と正解を0埋めし、下位の桁から並べる。"""
    a, b = map(int, problem)  # CALTのJSONLには文字列として保存される
    width = max(5, len(str(a)), len(str(b)))
    left = f"{a:0{width}d}"[::-1]
    right = f"{b:0{width}d}"[::-1]
    target = f"{int(answer):0{width + 1}d}"[::-1]
    return f"{left}+{right}=", target
