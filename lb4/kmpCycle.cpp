#include <iostream>
#include <string>
#include <vector>
#include <windows.h>



using namespace std;

// Функция для нахождения префикс-функции для строки Pattern (часть алгоритма КМП)
vector<int> prefixFunc(const string &Pattern) {
    cout << "   \n[ШАГ 1] Нахождение значения функции pi для строки: " << Pattern << endl;
    int p_len =(int)Pattern.size();
    vector<int> pi(p_len, 0);   // вектор pi, который будет хранить значения префикс-функции
    int j = 0;   //  длина текущего префикса, совпадающего с суффиксом
    for (int i = 1; i < p_len; ++i) {
        cout << "\n   Анализ символа '" << Pattern[i] << "' (позиция " << i << ") в шаблоне." << endl;
        while (j > 0 && Pattern[i] != Pattern[j]) {
            cout << "       Нашли различающиеся символы('" << Pattern[i] <<"' - позиция " << i <<", '" << Pattern[j] <<"' - позиция " << j << "),  возвращаемся к предыдущему индексу.\n";
            j = pi[j - 1];
            cout << "       После отката j = " << j << endl;
        }
        if (Pattern[i] == Pattern[j]) {
            cout << "       Символы совпадают ('" << Pattern[i] <<"' - позиция " << i <<", '" << Pattern[j] <<"' - позиция " << j << "), увеличиваем индекс j (количество символов с текущем найденном префиксе).\n";
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

// Алгоритм Кнута-Морриса-Пратта
int KMPcycle(const string& Pattern, const string& Text){
    cout << "======= Алгоритм Кнута-Морриса-Пратта. ========\n";
    int t_len = (int) Text.size(), p_len = (int)Pattern.size();
    vector<int> pi = prefixFunc(Pattern);
    cout << '\n';
    int ans = -1;
    int j = 0;
    cout << "\n\n[ШАГ 2] \n";
    cout << "Продолжаем алгоритм КМП. "
            "Проходим по удвоенной длине входного текста, далее"
            " берем значение индекса по модулю длины исходного текста\n";
     for (int i = 0; i < t_len * 2; ++i) {
         cout << "\n   Итерация i = " << i << ", символ в A: " << Text[i % t_len] << endl;
        while (j > 0 && Text[i%t_len] != Pattern[j]) {
            cout << "\n   Найдены отличающиеся символы в строке и подстроке"
                    "('" << Text[i%t_len] <<"' - позиция " << i%t_len <<" в А, '" << Pattern[j] <<"' - "
                      "позиция " << j << " в В),  возвращаемся к индексу"
                        ", который лежит в pi[" << j-1 << "]\n";
            j = pi[j - 1];
            cout << "   Индекс j после отката = " << j << endl;
        }
        if (Text[i%t_len] == Pattern[j]) {
            cout << "   Сравниваемые символы совпадают ('" << Text[i%t_len] <<"' - позиция " << i%t_len <<" "
                "в А, '" << Pattern[j] <<"' - позиция " << j << " в В),  производим поиск дальше. ";
            j++;
            cout << "\n   Текущий индекс увеличился j = " << j <<  endl;
        }
        if (j == p_len) {
            cout << "\n   Длина текущей найденной подстроки равна длине искомой строки,"
                    "возвращаем индекс, с которого началось совпадение - количество циклических сдвигов.\n";
            return (i - p_len + 1);
        }
    }
    return ans;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    cout << "Введите две строки A и B: ";
    string A, B;
    cin >> A >> B;

    if (A.size() != B.size()){
        cout << "Строки разных размеров, А - не циклический сдвиг В: ";
        cout << -1 << endl;
        return 0;
    }
    if (A == B){
        cout << "Строки идентичны; idx = " << 0 << endl;
        return 0;
    }

    int ans = KMPcycle(B, A);

    if(ans > A.size()){
        cout << "Число в ответе больше длины исходного текста. Ошибка, ответ - ";
        cout << -1 << endl;
        return 0;
    }

    cout << "Количество необходимых сдвигов - " << ans <<  endl;

    return 0;
}
