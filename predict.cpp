// Long short-term memory networks
// deep learning


#include <armadillo>
#include <stdexcept>
#include <string>
#include <GetOpt.h>
#include "network.cpp"


using namespace arma;
using namespace std;

int main(int argc, char** argv){

  string inputfile = "";
  double lr = 0.1;
  string networkfile = "";

  int option_char;
  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "i:n:")) != EOF)
	switch (option_char){
		case 'i': inputfile = optarg; break;
		case 'n': networkfile = optarg; break;
		// case 'o': output predictions flag
		case '?': cout << "usage: " << argv[0] << " -i input.csv -n network.bin" << endl;
		default: abort();
      }

	if(inputfile.length() == 0){
		cout << "Please name the file for data input." << endl;
		return 1;
	}
	if(networkfile.length() == 0){
		cout << "Please name the file to load the network." << endl;
		return 1;
	}

	cout << "Parameters: " << endl;
	cout << "____________" << endl;
	cout << "inputfile: " << inputfile << endl;
	cout << "networkfile: " << networkfile << endl;
	cout << endl;

	cout << "constructing network" << endl;
	Parameters params = Parameters(networkfile);
	RecurrentNetwork net = RecurrentNetwork(&params);

	cout << "Info: reading data from file" << endl;
	cout << "Info: no labels expected" << endl;
	cout << endl;
	// x values:
	mat inputvec;
	inputvec.load(inputfile);
	colvec targets;

	if(inputvec.n_cols > params.x_dim){
		cout << "discarding last column" << endl << endl;
		int targetcol = inputvec.n_cols-1;
		targets = inputvec.col(targetcol);
		inputvec.shed_col(targetcol);
	}
		
	cout << "presenting data to network" << endl;
	cout << "output:" << endl;
	for(int ind=0; ind<inputvec.n_rows; ind++){
		net.x_list_add(inputvec.row(ind).t());
		cout << net.nodelist[ind].state->h[0];
		if(targets.n_rows == inputvec.n_rows)
			cout << " | " << targets(ind) << endl;
		else
			cout << endl;
		net.x_list_clear();
	}
}

