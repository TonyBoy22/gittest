#pragma once

#include <string>
#include <map>
// Will define the person class and maybe other parameters

using namespace std;

class Person
{
public:
	// Needs
	std::string name;
	unsigned int type;	// The type of perso. Will be a dict
	float riskValue;
	float profit;
	// Couple of flags
	bool Resident;
	bool Maintainer;	// Wil be responsible for maintenance
	float PossessionFactor;

};

map <string, unsigned int> personType;
/*
* Will be let's say 0 for investors,
* 1 for roomates, 2 for associates workers
*/

