// Long short-term memory networks
// deep learning


#include <armadillo>
#include <stdexcept>
#include <string>
#include <getopt.h>
#include "network.cpp"


using namespace arma;
using namespace std;

int main(int argc, char** argv){

  string inputfile = "";
  string networkfile = "";
  int n_cells = 100;
  int iterations = 500;
  double lr = 0.1;
  string savefile = "";

  int option_char;
  // Invokes member function `int operator ()(void);'
  while ((option_char = getopt(argc, argv, "i:c:I:l:s:n:")) != EOF)
	switch (option_char){
		case 'i': inputfile = optarg; break;
		case 'c': n_cells = atoi(optarg); break;
		case 'I': iterations = atoi(optarg); break;
		case 'l': sscanf(optarg,"%lf",&lr);break;
		case 's': savefile = optarg; break;
		case 'n': networkfile = optarg; break;
		// case 'o': output predictions flag
		case '?': cout << "usage: " << argv[0] << " -i input.csv -c <cells> -I <iterations> -l <lr> -s save.bin" << endl;
		default: abort();
      }

	if(inputfile.length() == 0){
		cout << "Please name the file for data input." << endl;
		return 1;
	}

	cout << "Parameters: " << endl;
	cout << "____________" << endl;
	cout << "inputfile: " << inputfile << endl;
	cout << "ncells: " << n_cells << endl;
	cout << "iterations: " << iterations << endl;
	cout << "lr: " << lr << endl;
	if(savefile.length()>0)
		cout << "savefile: " << savefile << endl;
	else cout << "network will not be saved." << endl;
	cout << endl;

	cout << "Info: reading data from file" << endl;
	cout << "Info: labels expected in last column" << endl;
	// x values:
	mat inputvec;
	inputvec.load(inputfile); // autodetect file type
	// target values in last column
	int targetcol = inputvec.n_cols-1;
	colvec y_list(inputvec.col(targetcol));
	inputvec.shed_col(targetcol);
	int x_dim = inputvec.n_cols;
	cout << "x_dim: " << x_dim << endl;
	cout << endl;

	int n_layers = inputvec.n_rows;
	cout << "constructing network" << endl;
	QuadraticErrorLayer toyloss; //ClassErrorLayer 
	Parameters params = Parameters(n_cells, x_dim);
	if(networkfile.length()>0)
		params = Parameters(networkfile);
	RecurrentNetwork net = RecurrentNetwork(&params);

	cout << "starting training" << endl;
	for(int cur_iter=0; cur_iter<iterations; cur_iter++){
		for(int ind=0; ind<y_list.n_elem; ind++){ // 1 epoch
			net.x_list_add(inputvec.row(ind).t());
			//cout << net.nodelist[ind].state->h[0] << endl;
			//fprintf(stdout, "y_pred[%d] : %f, %f\n", ind, net.nodelist[net.nodelist.size()-1].state->h[0], y_list[ind]);
		}
		double loss = net.y_list_is(y_list, toyloss);
		params.apply_diff(lr);
		net.x_list_clear();
		//cout << "Iteration " << cur_iter << ". Loss: " << loss << endl;
	}
	cout << "training finished" << endl;

	cout << "saving network" << endl;
	if(savefile.length()>0)
		params.save(savefile); // binary file
}

