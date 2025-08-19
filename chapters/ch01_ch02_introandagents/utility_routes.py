#Why: Goals are binary; utilities encode graded preferences + uncertainty 

def expected_utility_route(base_time, delay_prob, delay_penalty, accident_risk, risk_weight=5.0):
    exp_time = base_time + delay_prob * delay_penalty
    util = -exp_time - risk_weight * accident_risk
    return util, exp_time

if __name__ == "__main__":
    A = expected_utility_route(10, 0.10, 8, 0.05, risk_weight=5.0)
    B = expected_utility_route(14, 0.01, 2, 0.01, risk_weight=5.0)
    print(f"Route A -> utility={A[0]:.2f}, expected_time={A[1]:.2f}")
    print(f"Route B -> utility={B[0]:.2f}, expected_time={B[1]:.2f}")

# Route A is faster but slightly riskier.
# Route B is safer but slower.