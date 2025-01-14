pub fn rosenbrock(x: &[f64; 2], a: f64, b: f64) -> f64 {
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
}

pub fn rosenbrock_grad(x: &[f64; 2], a: f64, b: f64) -> [f64; 2] {
    let g_0 = 4. * b * x[0].powi(3) + (2. - 4. * b * x[1]) * x[0] - 2. * a;
    let g_1 = 2. * b * (x[1] - x[0].powi(2));
    [g_0, g_1]
}
