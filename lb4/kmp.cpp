#include <iostream>
#include <string>
#include <vector>
#include <windows.h>

using namespace std;

vector<int> prefixFunc(const string &P) {
    cout << "   \n[ШАГ 1] Нахождение значения функции pi для строки: " << P << endl;
    int p_len =(int)P.size();
    vector<int> pi(p_len, 0);
    int j = 0;
    cout << "\n   pi[0] = 0 независимо от строки\n";
    for (int i = 1; i < p_len; i++) {
        cout << "\n   Анализ символа '" << P[i] << "' (позиция " << i << ") в шаблоне." << endl;
        while (j > 0 && P[i] != P[j]) {
            cout << "       Нашли различающиеся символы('" << P[i] <<"' - позиция " << i <<", '" << P[j] <<"' - позиция " << j << "),  возвращаемся к предыдущему индексу.\n";
            j = pi[j - 1];
            cout << "       После отката j = " << j << endl;
        }
        if (P[i] == P[j]) {
            cout << "       Символы совпадают ('" << P[i] <<"' - позиция " << i <<", '" << P[j] <<"' - позиция " << j << "), увеличиваем индекс j (количество символов с текущем найденном префиксе).\n";
            j++;
            cout << "       j = " << j << endl;

        }
        cout << "       Присваиваем значение индекса j в pi[" << i << "] = " << j << endl;
        pi[i] = j;
    }
    cout << "   \n[РЕЗУЛЬТАТ] Префикс-функция вычислена: ";
    for (int val : pi) cout << val << " ";
    cout << endl;
    return pi;
}

vector<int> KMT(string P, string T){
    cout << "======= Алгоритм Кнута-Морриса-Пратта. ========\n";
    int t_len = (int) T.size(), p_len = (int)P.size();
    vector<int> pi = prefixFunc(P);
    vector<int> ans;
    cout << '\n';
    int j = 0;
    cout << "\n[ШАГ 2] ";
    cout << "Продолжаем алгоритм КМП. \n";
    for (int i = 0; i < t_len; i++) {
        cout << "   Итерация i = " << i << ", символ в строке поиска: " << T[i % t_len] << endl;
        while (j > 0 && T[i] != P[j]) {
            cout << "\n   Найдены отличающиеся символы в строке и подстроке"
                    "('" << T[i] <<"' - позиция " << i <<", '" << P[j] <<"' - позиция " << j << "),  возвращаемся к индексу"
                    ", который лежит в pi[" << j-1 << "]\n";
            j = pi[j - 1];
            cout << "   Индекс j после отката = " << j << "\n\n";
        }
        if (T[i] == P[j]) {
            cout << "   Сравниваемые символы совпадают ('" << T[i] <<"' - позиция " << i <<", '" << P[j] <<"' - позиция " << j << "),  производим поиск дальше. ";
            j++;
            cout << "Текущий индекс подстроки j - " << j <<  endl;
        }
        if (j == p_len) {
            cout << "\n     Длина входной подстроки совпала с найденной подстрокой, "
                    "записываем индекс начала подстроки в тексте в результат.\n";
            ans.push_back(i - p_len + 1);
            j = pi[j - 1];
            cout << "   Новое j = " << j << " (pi[" <<j-1 <<"])\n\n";
        }
    }
    if(ans.empty()){
        cout << "[РЕЗУЛЬТАТ] Подстрока не обнаружена в тексте. [-1]\n";
        return {-1};
    }

    return ans;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    cout << "Введите подстроку и строку, в которой будет производиться поиск: ";
    string P;
    string T;
    cin >> P;
    cin >> T;
    vector<int> res = KMT(P, T);
    if(res[0] != -1) {
        cout << "\n[РЕЗУЛЬТАТ] Элементы результирующего вектора: ";
        for (size_t i = 0; i < res.size(); i++) {
            if (i > 0) cout << ",";
            cout << res[i];
        }
        cout << endl;
    }
    return 0;
}