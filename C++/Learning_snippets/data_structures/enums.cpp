#include "enums.h"
#include <iostream>

enum enumChar : char {
	first, second = 'z', next, ohComeOn, okStopItNow
};
/* enum counts only if value type is int or if a value has been assigned
* meaning that in this case, second and after have a letter value increasing in
* ASCII table order. Those before not have value
* 
* for docs as how to split it between .h and .cpp
* https://stackoverflow.com/questions/1284529/enums-can-they-do-in-h-or-must-stay-in-cpp
* 
*/
int main() {
	enumChar var = next;
	std::cout << "next char value is " << var << std::endl;
}