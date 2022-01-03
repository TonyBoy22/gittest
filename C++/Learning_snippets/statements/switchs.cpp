#include <iostream>

using namespace std;

void basic_switch() {
	/*
	makes a basic swithc statement with an input and 
	an according output
	*/
	cout << "What do you do\n";
	char answer;
	cin >> answer;

	switch (answer) {
	case 'a':
		cout << "answer was a";
		break;
	case 'b':
		break;
	case 'c':
		cout << "answer was b or c";
		break;
	default:
		cout << "anything else";
		break;
	}
	
}


//int main() {
//	basic_switch();
//}