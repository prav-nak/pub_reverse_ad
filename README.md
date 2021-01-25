Automatic differentiation computes derivative values of computer implemented functions with accuracy up to machine precision.
The main principle behind automatic differentiation is that every function can be represented as a finite sequence of elemental unary or binary operations or static single assignment (SSA) with known derivatives e.g. unary operations as trigonometric, exponential and logarithmic operations or binary operations as addition, multiplication, division and the power operation. The derivative of the whole computation can then be computed through the chain rule, something along the lines of: 


```cpp
struct BinaryExpr;
struct BaseExpr;
struct AddExpr;
struct MulExpr;

using BaseExprPtr = std::shared_ptr<BaseExpr>;
using DerivativesMap = std::unordered_map<const BaseExpr*, double>;

struct BaseExpr{

	BaseExpr() = delete;
	explicit BaseExpr(double _val) : val(_val) {}

	virtual void back_propagate(DerivativesMap&, double wprime) const = 0;

	double val;
};

BaseExprPtr operator+(const BaseExprPtr& r);
BaseExprPtr operator+(const BaseExprPtr& l, const BaseExprPtr& r); 
BaseExprPtr operator*(const BaseExprPtr& l, const BaseExprPtr& r); 

struct UnaryExpr : BaseExpr {
	BaseExprPtr x;
	UnaryExpr(double val, const BaseExprPtr& _x) : BaseExpr(val), x(_x) {}
};

struct BinaryExpr : BaseExpr {

	BaseExprPtr l, r;
	BinaryExpr(double val, const BaseExprPtr& _l, const BaseExprPtr& _r) : BaseExpr(val), l(_l), r(_r) {}

};

struct AddExpr : BinaryExpr {
	using BinaryExpr::BinaryExpr;
	virtual void back_propagate(DerivativesMap& derivatives, double wprime) const {
		l->back_propagate(derivatives, wprime);
		r->back_propagate(derivatives, wprime);
	}
};

struct MulExpr : BinaryExpr {
	using BinaryExpr::BinaryExpr;
	virtual void back_propagate(DerivativesMap& derivatives, double wprime) const {
		l->back_propagate(derivatives, wprime*r->val);
		r->back_propagate(derivatives, wprime*l->val);
	}
};

struct TerminalExpr : BaseExpr {
	using BaseExpr::BaseExpr;
	virtual void back_propagate(DerivativesMap& derivatives, double wprime) const {
		auto it = derivatives.find(this);
		if (it != derivatives.end()) {
			it->second += wprime;
		}
		else {
			derivatives.insert({this, wprime});
		}
	}
};

struct VariableExpr : BaseExpr {
	BaseExprPtr expr;
	VariableExpr() = delete;

	explicit VariableExpr(const BaseExprPtr& _expr) : BaseExpr(_expr->val), expr(_expr){}

	virtual void back_propagate(DerivativesMap& derivatives, double wprime) const {
		const auto it = derivatives.find(this);
		if (it != derivatives.end()) {
			it->second += wprime;
		}
		else {
			derivatives.insert({this, wprime});
		}
		expr->back_propagate(derivatives, wprime);
	}
};

struct var {
	BaseExprPtr expr;
	var() = delete;
	explicit var(double _val) : expr(std::make_shared<TerminalExpr>(_val)){ }
	var(const BaseExprPtr& _expr): expr(std::make_shared<VariableExpr>(_expr)){ }
};

BaseExprPtr operator+(const BaseExprPtr& r){ return r; }
BaseExprPtr operator+(const BaseExprPtr& l, const BaseExprPtr& r){ 
	return std::make_shared<AddExpr>(l->val+r->val, l, r);
}
BaseExprPtr operator*(const BaseExprPtr& l, const BaseExprPtr& r){ 
	return std::make_shared<MulExpr>(l->val*r->val, l, r);
}
```


*Like other repos, this one is encrypted using git crypt. If you would like to collaborate please contact.*
