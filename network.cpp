// Long short-term memory networks
// deep learning


#include <armadillo>
#include <stdexcept>
#include <string>
#include <vector>
#include <assert.h>
#include <sstream>
#include <getopt.h>


using namespace arma;
using namespace std;

mat sigmoid(mat A){
	return 1.0/(1 + exp(-A));
}


class Parameters{
	public:
		Parameters(int n_cells, int x_dim);
		Parameters(string filename);
		void save(string filename);
		int n_cells, x_dim;
		void init();
		mat init_term(double a, double b, int rows, int cols);
		colvec init_term(double a, double b, int rows);
		void apply_diff(double lr=0.1);

		mat wg, wi, wf, wo; // weight matrices
		colvec bg, bi, bf, bo; // bias terms
		mat wg_diff, wi_diff, wf_diff, wo_diff, 
			bg_diff, bi_diff, bf_diff, bo_diff; // derivative of loss function
};

Parameters::Parameters(string filename){
	cout << "clearing network weights and biases" << endl;
	cout << "loading weights from " << filename << endl;
	field<mat> weights;
	weights.load(filename);

	this->n_cells = weights(0,0).n_rows;
	int concat_len = weights(0,0).n_cols;
	this->x_dim = concat_len - n_cells;
	this->init();

	// weight matrices
	this->wg = weights(0,0);
	this->wi = weights(0,1);
	this->wf = weights(0,2);
	this->wo = weights(0,3);
	
	// bias terms
	this->bg = weights(1,0);
	this->bi = weights(1,1);
	this->bf = weights(1,2);
	this->bo = weights(1,3);

	cout << "weights loaded." << endl;
}

void Parameters::save(string filename){
	field<mat> weights(2,4);
	// weight matrices
	weights(0,0) = this->wg;
	weights(0,1) = this->wi;
	weights(0,2) = this->wf;
	weights(0,3) = this->wo;
	// bias terms
	weights(1,0) = this->bg;
	weights(1,1) = this->bi;
	weights(1,2) = this->bf;
	weights(1,3) = this->bo;

	weights.save(filename);
	cout << "weights saved." << endl;
}

mat Parameters::init_term(double a, double b, int rows, int cols){
	mat B(rows, cols, fill::randu);
	B = (b - a) * B + a;
	return B;
}
colvec Parameters::init_term(double a, double b, int rows){
	colvec B(rows, fill::randu);
	B = (b - a) * B + a;
	return B;
}

void Parameters::apply_diff(double lr){
	this->wg -= lr * this->wg_diff;
	this->wi -= lr * this->wi_diff;
	this->wf -= lr * this->wf_diff;
	this->wo -= lr * this->wo_diff;
	this->bg -= lr * this->bg_diff;
	this->bi -= lr * this->bi_diff;
	this->bf -= lr * this->bf_diff;
	this->bo -= lr * this->bo_diff;

	// useless reset? 
	this->wg_diff.zeros(size(this->wg));
	this->wi_diff.zeros(size(this->wi));
	this->wf_diff.zeros(size(this->wf));
	this->wo_diff.zeros(size(this->wo));
	this->bg_diff.zeros(size(this->bg));
	this->bi_diff.zeros(size(this->bi));
	this->bf_diff.zeros(size(this->bf));
	this->bo_diff.zeros(size(this->bo));
}

void Parameters::init(){
	arma_rng::set_seed_random();
	int concat_len = x_dim + n_cells;
	// init weights:
	this->wg = init_term(-0.1, 0.1, this->n_cells, concat_len);
	this->wi = init_term(-0.1, 0.1, this->n_cells, concat_len);
	this->wf = init_term(-0.1, 0.1, this->n_cells, concat_len);
	this->wo = init_term(-0.1, 0.1, this->n_cells, concat_len);

	// init biases:
	this->bg = init_term(-0.1, 0.1, this->n_cells);
	this->bi = init_term(-0.1, 0.1, this->n_cells);
	this->bf = init_term(-0.1, 0.1, this->n_cells);
	this->bo = init_term(-0.1, 0.1, this->n_cells);

	// init derivatives:
	this->wg_diff.zeros(n_cells, concat_len);
	this->wi_diff.zeros(n_cells, concat_len);
	this->wf_diff.zeros(n_cells, concat_len);
	this->wo_diff.zeros(n_cells, concat_len);
	this->bg_diff.zeros(n_cells);
	this->bi_diff.zeros(n_cells);
	this->bf_diff.zeros(n_cells);
	this->bo_diff.zeros(n_cells);
}

Parameters::Parameters(int n_cells, int x_dim){
	this->n_cells = n_cells;
	this->x_dim = x_dim;
	init();
}

class NetworkState{
	public:
		NetworkState(int n_cells, int x_dim);
		colvec g, i, f, o, s, 
			h, bottom_diff_h, bottom_diff_s, bottom_diff_x;
};

NetworkState::NetworkState(int n_cells, int x_dim){
        this->g.zeros(n_cells);
        this->i.zeros(n_cells);
        this->f.zeros(n_cells);
        this->o.zeros(n_cells);
        this->s.zeros(n_cells);
        this->h.zeros(n_cells);
        this->bottom_diff_h.zeros(size(this->h));
        this->bottom_diff_s.zeros(size(this->s));
        this->bottom_diff_x.zeros(x_dim);
}


class NetworkNode{
	public:
		NetworkNode(Parameters *params);
		Parameters *params;
		NetworkState *state;
		void bottom_data_is(colvec x, colvec s_prev, colvec h_prev);
		void bottom_data_is(colvec x);
		void top_diff_is(colvec top_diff_h, colvec top_diff_s);

		colvec x; // non-recurrent input to node
		colvec xc; // non-recurrent input concatenated with recurrent input
		colvec s_prev, h_prev;
		colvec h, s;
};

NetworkNode::NetworkNode(Parameters *params){
	// store reference to parameters and to activations:
	this->params = params;
	this->state = new NetworkState(this->params->n_cells, this->params->x_dim);

}

void NetworkNode::bottom_data_is(colvec x){
    // if this is the first node in the network
	colvec s_prev, h_prev;
	s_prev.zeros(size(this->state->s));
	h_prev.zeros(size(this->state->h));
	this->bottom_data_is(x, s_prev, h_prev);
}

void NetworkNode::bottom_data_is(colvec x, colvec s_prev, colvec h_prev){
	// save data for use in backprop;
	this->s_prev = s_prev;
	this->h_prev = h_prev;
	// concatenate x(t) and h(t-1);
	colvec xc = join_cols(x, h_prev);
	this->state->g = tanh(this->params->wg * xc + this->params->bg);
	this->state->i = sigmoid(this->params->wi * xc + this->params->bi);
	this->state->f = sigmoid(this->params->wf * xc + this->params->bf);
	this->state->o = sigmoid(this->params->wo * xc + this->params->bo);
	this->state->s = this->state->g % this->state->i + s_prev % this->state->f;
	this->state->h = this->state->s % this->state->o;
	this->x = x;
	this->xc = xc;
}


void NetworkNode::top_diff_is(colvec top_diff_h, colvec top_diff_s){
	// notice that top_diff_s is carried along the constant error carousel
	colvec ds = this->state->o % top_diff_h + top_diff_s;
	colvec vdo = this->state->s % top_diff_h;
	colvec di = this->state->g % ds;
	colvec dg = this->state->i % ds;
	colvec df = this->s_prev % ds;

	// diffs w.r.t. vector inside sigmoid / tanh function
	colvec di_input = (1.0 - this->state->i) % this->state->i % di;
	colvec df_input = (1.0 - this->state->f) % this->state->f % df;
	colvec do_input = (1.0 - this->state->o) % this->state->o % vdo;
	colvec dg_input = (1.0 - pow(this->state->g, 2)) % dg;

	// diffs w->r->t-> inputs
	this->params->wi_diff += di_input * this->xc.t();
	this->params->wf_diff += df_input * this->xc.t();
	this->params->wo_diff += do_input * this->xc.t();
	this->params->wg_diff += dg_input * this->xc.t();
	this->params->bi_diff += di_input;
	this->params->bf_diff += df_input;
	this->params->bo_diff += do_input;
	this->params->bg_diff += dg_input;

	// compute bottom diff
	colvec dxc;
	dxc.zeros(size(this->xc));
	dxc += this->params->wi.t() * di_input;
	dxc += this->params->wf.t() * df_input;
	dxc += this->params->wo.t() * do_input;
	dxc += this->params->wg.t() * dg_input;

	// save bottom diffs
	this->state->bottom_diff_s = ds % this->state->f;
	this->state->bottom_diff_x = dxc.head(this->params->x_dim);
	this->state->bottom_diff_h = dxc.tail(this->state->h.n_elem);
}

class LossLayer{
	public:
		virtual double loss(colvec pred, double label) = 0;
		virtual colvec bottom_diff(colvec pred, double label) = 0;
 };


class QuadraticErrorLayer: public LossLayer{
    // Computes square loss with first element of hidden layer array.
	public:
		double loss(colvec pred, double label);
		colvec bottom_diff(colvec pred, double label);
};

double QuadraticErrorLayer::loss(colvec pred, double label){
	return pow(pred(0) - label, 2);
}

colvec QuadraticErrorLayer::bottom_diff(colvec pred, double label){
	colvec diff;
	diff.zeros(size(pred));
	diff(0) = 2 * (pred(0) - label);
	return diff;
}

class ClassErrorLayer: public LossLayer{
    // Computes square loss with first element of hidden layer array.
	public:
		double loss(colvec pred, double label);
		colvec bottom_diff(colvec pred, double label);
};

double ClassErrorLayer::loss(colvec pred, double label){
	return 1 * (pred(0) != label);
}

colvec ClassErrorLayer::bottom_diff(colvec pred, double label){
	colvec diff;
	diff.zeros(size(pred));
	diff(0) = 1 * (pred(0) != label);
	return diff;
}



class RecurrentNetwork{
	public:
		RecurrentNetwork(Parameters *params);
		double y_list_is(colvec y_list, LossLayer &loss_layer);
		void x_list_clear();
		void x_list_add(colvec x);

		Parameters *params;
		vector<NetworkNode> nodelist;
		vector<colvec> x_list; // inputs
};

RecurrentNetwork::RecurrentNetwork(Parameters *params){
	this->params = params;
	this->nodelist.clear();
	this->x_list.clear();
}


double RecurrentNetwork::y_list_is(colvec y_list, LossLayer &loss_layer){
	/*
	Updates diffs by setting target sequence 
	with corresponding loss layer. 
	Will *NOT* update parameters.  To update parameters,
	call this->params->apply_diff()
	*/
	assert(y_list.n_elem == this->x_list.size());
	int idx = this->x_list.size() - 1;
	// first node only gets diffs from label
	double loss = loss_layer.loss(this->nodelist[idx].state->h, y_list[idx]);
	colvec diff_h = loss_layer.bottom_diff(this->nodelist[idx].state->h, y_list[idx]);
	// here s is not affecting loss due to h(t+1), hence we set equal to zero
	colvec diff_s;
	diff_s.zeros(params->n_cells);
	this->nodelist[idx].top_diff_is(diff_h, diff_s);

	idx -= 1;
	// ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
	// we also propagate error along constant error carousel using diff_s
	while(idx >= 0){
		loss += loss_layer.loss(this->nodelist[idx].state->h, y_list[idx]);
		diff_h = loss_layer.bottom_diff(this->nodelist[idx].state->h, y_list[idx]);
		diff_h += this->nodelist[idx + 1].state->bottom_diff_h;
		diff_s = this->nodelist[idx + 1].state->bottom_diff_s;
		this->nodelist[idx].top_diff_is(diff_h, diff_s);
		idx -= 1;
	}
	return loss;
}

void RecurrentNetwork::x_list_clear(){
	this->x_list.clear();
}

void RecurrentNetwork::x_list_add(colvec x){
	this->x_list.push_back(x);
	if(this->x_list.size() > this->nodelist.size()){
		// need to add new network node, create new state mem
		this->nodelist.push_back(NetworkNode(this->params));
	}

	// get index of most recent x input
	int idx = this->x_list.size() - 1;
	if(idx == 0){
		// no recurrent inputs yet;
		this->nodelist[idx].bottom_data_is(x);
	} else{
		colvec s_prev = this->nodelist[idx - 1].state->s;
		colvec h_prev = this->nodelist[idx - 1].state->h;
		this->nodelist[idx].bottom_data_is(x, s_prev, h_prev);
	}
}

