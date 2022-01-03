#include <iostream>
#include <vector>

using namespace std;

/*
trying stuff on vectors
*/
// if wrong project launched, switch to set as startup project
int main() {
	vector<char> v1 = { 'a', 'n', 'a' };
	cout << "Vector declared is "; // << v1 << endl;
	// Cannot print a full vector in this situation, << operator 
	// are not recognized. Would work if it was an int or 1 element var?
}