//#include "string_streams.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;
// goal is to create a parse function

// input a string with numbers and parse it to extract int
 std::vector<int> ParseStr(const std::string str) {
	 std::vector<int> v;
	 std::stringstream sso(str);
	 char ch;
	 int tmp; 
	 while (sso >> tmp) {
		 // tant que l'opération ne donne pas faux, soit tan
		 // qu'il y a des chiffres à sortir
		 v.push_back(tmp);
		 // Sort la virgule de la string
		 sso >> ch;
		 // sso se base sur le type de la variable dans laquelle mettre
		 // la valeur pour savori quelle quantité de données y mettre
		 // selon le type de ch, si c'Est string par exeple, sso va y mettre toute la string
	 }

	 return v;
}

 /*
 Other ressource for sstream
 https://linuxhint.com/c-stringstream-and-how-to-use-it/

 */

 int main() {
	 // basically
	 string input = "23,26,5";

	 vector<int> v = ParseStr(input);
	 int s = v.size();
	 for (int& value : v) {
		 cout << value << endl;
	 }
	
 }