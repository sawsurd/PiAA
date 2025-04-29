def wagner_fischer(s1, s2, replace_cost, insert_cost, delete_cost, double_delete_cost) -> tuple:
    print("==== Алгоритм Вагнера-Фишера ====")
    m, n = len(s1), len(s2)
    print("Длина первой строки -", m, "\nДлина второй строки -", n)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    op = [[None] * (n + 1) for _ in range(m + 1)]
    print('=' * 100)
    print(f"Стоимость вставки = {insert_cost}, стоимость удаления = {delete_cost},"
          f" стоимость замены = {replace_cost}, стомость двойного удаления = {double_delete_cost}")
    print('=' * 100)
    print("Заполняем ячейку [0][0] нулем")
    d[0][0] = 0

    print("Заполняем первый столбец и первую строку")
    for j in range(1, n + 1):
        d[0][j] = d[0][j - 1] + insert_cost
        print(f"d[0][{j}] = {d[0][j]} (вставка {insert_cost})")
        op[0][j] = 'I'

    print('=' * 100)

    for i in range(1, m + 1):
        d[i][0] = d[i - 1][0] + delete_cost
        op[i][0] = 'D'
        if i >= 2 and s1[i - 2] != s1[i - 1]:
            if d[i][0] > d[i - 2][0] + double_delete_cost:
                d[i][0] = d[i - 2][0] + double_delete_cost
                op[i][0] = 'O'
        print(f"d[{i}][0] = {d[i][0]} (удаление {delete_cost})")
    print('=' * 100)

    print("Заполняем остальные ячейки матрицы")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                print(f"Значения совпали (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                      f" d[{i}][{j}] = {d[i][j]} ")
                op[i][j] = 'M'
            else:
                replace = d[i - 1][j - 1] + replace_cost
                insert = d[i][j - 1] + insert_cost
                delete = d[i - 1][j] + delete_cost
                double_delete = float('inf')
                if i >= 2 and s1[i - 2] != s1[i - 1]:
                    double_delete = d[i - 2][j] + double_delete_cost
                print("Выбираем менее дорогостоящую операцию")
                print(f"replace - {replace}, insert - {insert}, delete - {delete}, ddelete - {double_delete}")
                d[i][j] = min(replace, insert, delete, double_delete)

                if d[i][j] == replace:
                    print(f"Была выбрана замена (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                          f" d[{i}][{j}] = {d[i][j]}")
                    op[i][j] = 'R'
                elif d[i][j] == insert:
                    print(f"Была выбрана вставка (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                          f" d[{i}][{j}] = {d[i][j]}")
                    op[i][j] = 'I'
                elif d[i][j] == delete:
                    print(
                        f"Было выбрано удаление 1 символа (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                        f" d[{i}][{j}] = {d[i][j]}")
                    op[i][j] = 'D'

                else:
                    print(
                        f"Было выбрано удаление 2-х отличающихся символов (s1[{i - 1}] = {s1[i - 1]}),"
                        f" (s1[{i - 2}] = {s1[i - 2]}), (s2[{j - 1}] = {s2[j - 1]}), d[{i}][{j}] = {d[i][j]}")
                    op[i][j] = 'O'
    print('=' * 100)
    
    print_table(d, s1, s2)
    return d, op


def print_table(dp, s1, s2) -> None:
    n = len(s1)
    m = len(s2)
    max_val = max(max(row) for row in dp)
    cell_width = max(3, len(str(max_val)) + 2)

    header = [" "] + [" "] + list(s2)
    print("-" + ("-" * (cell_width + 2) + "-") * (m + 2))

    header_row = "|"
    for h in header:
        header_row += f" {h:^{cell_width}} |"
    print(header_row)

    print("-" + ("-" * (cell_width + 2) + "-") * (m + 2))

    for i in range(n + 1):
        row_header = " " if i == 0 else s1[i - 1]
        row = [f" {row_header:^{cell_width}} |"]
        for j in range(m + 1):
            cell = dp[i][j]
            cell_str = f"{cell:^{cell_width}}"
            row.append(f" {cell_str} |")
        print("|" + "".join(row))
        print("-" + ("-" * (cell_width + 2) + "-") * (m + 2))


def reconstruct_operations(op, s1, s2) -> str:
    operations = []
    i, j = len(s1), len(s2)
    print('=' * 100)
    print("\nВосстановление последовательности операций:")
    while i > 0 or j > 0:
        current_op = op[i][j]
        operations.append(current_op)
        print(f"Операция {current_op} на позиции ({i}, {j})")
        if current_op == 'M' or current_op == 'R':
            i -= 1
            j -= 1
        elif current_op == 'I':
            j -= 1
        elif current_op == 'D':
            i -= 1
        elif current_op == 'O':
            i -= 2

    operations.reverse()
    return ''.join(operations)


def levenshtein_distance(s1, s2) -> int:
    m, n = len(s1), len(s2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    print("\nВычисление расстояния Левенштейна без двойного удаления:")
    d[0][0] = 0
    print('=' * 100)
    print("Заполнение первого столбца и первой строки")
    for j in range(1, n + 1):
        d[0][j] = d[0][j - 1] + 1
        print(f"d[0][{j}] = {d[0][j]}")

    for i in range(1, m + 1):
        d[i][0] = d[i - 1][0] + 1
        print(f"d[{i}][0] = {d[i][0]}")
    print('=' * 100)
    print("Заполнение остальных ячеек")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                print(f"Значения совпали (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}), "
                      f"d[{i}][{j}] = {d[i][j]} ")
                d[i][j] = d[i - 1][j - 1]
            else:
                replace = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                print("Выбираем менее дорогостояющую операцию")
                print(f"replace - {replace}, insert - {insert}, delete - {delete}")
                d[i][j] = min(replace, insert, delete)
                if d[i][j] == replace:
                    print(f"Была выбрана замена (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = "
                          f"{s2[j - 1]}), d[{i}][{j}] = {d[i][j]}")
                elif d[i][j] == insert:
                    print(f"Была выбрана вставка (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                          f" d[{i}][{j}] = {d[i][j]}")
                elif d[i][j] == delete:
                    print(f"Было выбрано удаление 1 символа (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}), "
                          f"d[{i}][{j}] = {d[i][j]}")
    print('=' * 100)
    print_table(d, s1, s2)

    return d[m][n]


def levenshtein_double(s1, s2) -> int:
    m, n = len(s1), len(s2)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    print("\nВычисление расстояния Левенштейна с двойным удалением:")
    print('=' * 100)
    print("Заполнение первого столбца и первой строки")
    for i in range(1, m + 1):
        d[i][0] = d[i - 1][0] + 1
        if i >= 2 and s1[i - 2] != s1[i - 1]:
            if d[i][0] > d[i - 2][0] + 1:
                d[i][0] = d[i - 2][0] + 1
        print(f"d[{i}][0] = {d[i][0]}")

    for j in range(1, n + 1):
        d[0][j] = d[0][j - 1] + 1
        print(f"d[0][{j}] = {d[0][j]}")
    print('=' * 100)
    print("Заполнение остальных ячеек")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                print(f"Значения совпали (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                      f" d[{i}][{j}] = {d[i][j]} ")
            else:
                replace = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                double_delete = float('inf')
                if i >= 2 and s1[i - 2] != s1[i - 1]:
                    double_delete = d[i - 2][j] + 1
                print("Выбираем менее дорогостоящую операцию")
                print(f"replace - {replace}, insert - {insert}, delete - {delete}, ddelete - {double_delete}")
                d[i][j] = min(replace, insert, delete, double_delete)
                if d[i][j] == replace:
                    print(f"Была выбрана замена (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}),"
                          f" d[{i}][{j}] = {d[i][j]}")
                elif d[i][j] == insert:
                    print(f"Была выбрана вставка (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}), "
                          f"d[{i}][{j}] = {d[i][j]}")
                elif d[i][j] == delete:
                    print(f"Было выбрано удаление 1 символа (s1[{i - 1}] = {s1[i - 1]}), (s2[{j - 1}] = {s2[j - 1]}), "
                          f"d[{i}][{j}] = {d[i][j]}")
                else:
                    print(f"Было выбрано удаление 2-х отличающихся символов (s1[{i - 1}] = {s1[i - 1]}),"
                          f" (s1[{i - 2}] = {s1[i - 2]}), (s2[{j - 1}] = {s2[j - 1]}), d[{i}][{j}] = {d[i][j]}")
    print('=' * 100)
    print_table(d, s1, s2)
    return d[m][n]


def main() -> None:
    replace_cost, insert_cost, delete_cost, double_delete_cost = map(int, input(
        "Введите стоимость Replace, Insert, Delete, DoubleDelete через пробел: ").split())
    s1 = input("Введите первую строку: ").strip()
    s2 = input("Введите вторую строку: ").strip()

    d, op = wagner_fischer(s1, s2, replace_cost, insert_cost, delete_cost, double_delete_cost)

    operations_sequence = reconstruct_operations(op, s1, s2)

    lev_distance = levenshtein_distance(s1, s2)
    lev_distance2 = levenshtein_double(s1, s2)
    print('=' * 100)
    print("\nРезультаты:")
    print("Алгоритм Вагнера-Фишера:", d[-1][-1])
    print("Последовательность операций:", operations_sequence)
    print(s1)
    print(s2)
    print("Расстояние Левенштейна (без двойного удаления):", lev_distance)
    print("Расстояние Левенштейна (с двойным удалением):", lev_distance2)


if __name__ == "__main__":
    main()
