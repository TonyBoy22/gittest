#include <iostream>
#include <vector>

using namespace std;

/*
different style of loops and iterate over containers
*/

void range_for_loop() {
	// set an iterable
	vector<int> v1 = { 0, 1, 3, 4 };

	// Could assign auto, but we know v1 contains ints
	for (int x : v1) {
		cout << "Value in v1 is " << x << endl;
	}
}

int main() {
	range_for_loop();
}