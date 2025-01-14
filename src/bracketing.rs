pub fn bracket_minimum(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    s: f64,
    k: f64,
    iter_limit: usize,
) -> Option<(f64, f64)> {
    assert!(k > 1.);
    assert!(s > 0.);

    let (mut a, ya) = (x0, f(x0));
    let (mut b, mut yb) = (a + s, f(a + s));
    let mut s = s;
    let mut iter_limit = iter_limit;

    if yb > ya {
        (a, b) = (b, a);
        yb = ya;
        s = -s;
    }

    while iter_limit > 0 {
        let (c, yc) = (b + s, f(b + s));
        if yc > yb {
            return if a < c { Some((a, c)) } else { Some((c, a)) };
        }
        (a, b, yb) = (b, c, yc);
        s *= k;

        iter_limit -= 1;
    }

    None
}

pub fn fibonacci_search(
    f: &dyn Fn(f64) -> f64,
    bracket: (f64, f64),
    search_count: usize,
    eps: f64,
) -> (f64, f64) {
    let (mut a, mut b) = bracket;
    assert!(a < b);

    let s = (1. - 5f64.sqrt()) / (1. + 5f64.sqrt());
    let phi = (1. + 5f64.sqrt()) / 2.;
    let rho = |i: usize| (1. - s.powi(i as i32)) / (phi * (1. - s.powi(i as i32 + 1)));

    let mut d = rho(search_count) * b + (1. - rho(search_count)) * a;
    let mut yd = f(d);

    for i in 1..search_count {
        let c = if i == search_count - 1 {
            eps * a + (1. - eps) * d
        } else {
            rho(search_count - i + 1) * a + (1. - rho(search_count - i + 1)) * b
        };
        let yc = f(c);
        if yc < yd {
            (b, d, yd) = (d, c, yc);
        } else {
            (a, b) = (b, c);
        }
    }

    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub fn golden_section_search(
    f: &dyn Fn(f64) -> f64,
    bracket: (f64, f64),
    search_count: usize,
) -> (f64, f64) {
    let (mut a, mut b) = bracket;
    let phi = (1. + 5f64.sqrt()) / 2.;
    let rho = phi - 1.;
    let mut d = rho * b + (1. - rho) * a;
    let mut yd = f(d);

    for _ in 1..search_count {
        let c = rho * a + (1. - rho) * b;
        let yc = f(c);
        if yc < yd {
            (b, d, yd) = (d, c, yc);
        } else {
            (a, b) = (b, c);
        }
    }

    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bracket_minimum_0() {
        let res = bracket_minimum(&|x| 3. * x * x - x + 5., -4., 1e-2, 2., 100);
        assert!(res.is_some());
        let bracket = res.unwrap();
        let min_x = 1. / 6.;
        assert!(bracket.0 < min_x);
        assert!(bracket.1 > min_x);
    }

    #[test]
    fn test_bracket_minimum_1() {
        let res = bracket_minimum(&|x| (-x).exp(), -4., 1e-2, 2., 100);
        assert!(res.is_none());
    }

    #[test]
    fn test_fabonacci_search_0() {
        let f = |x| 3. * x * x - x + 5.;
        let res = bracket_minimum(&f, -4., 1e-2, 2., 100);
        assert!(res.is_some());
        let bracket = res.unwrap();
        println!("initial bracket is {:?}", bracket);
        let bracket = fibonacci_search(&f, bracket, 24, 0.01);
        let min_x = 1. / 6.;
        assert!(bracket.0 <= min_x);
        assert!(
            min_x - bracket.0 < 1e-4,
            "min_x - bracket.0 = {}",
            min_x - bracket.0
        );
        assert!(bracket.1 >= min_x);
        assert!(
            bracket.1 - min_x < 1e-4,
            "bracket.1 - min_x = {}",
            bracket.1 - min_x
        );
    }

    #[test]
    fn test_golden_section_search_0() {
        let f = |x| 3. * x * x - x + 5.;
        let res = bracket_minimum(&f, -4., 1e-2, 2., 100);
        assert!(res.is_some());
        let bracket = res.unwrap();
        println!("initial bracket is {:?}", bracket);
        let bracket = golden_section_search(&f, bracket, 24);
        let min_x = 1. / 6.;
        assert!(bracket.0 <= min_x);
        assert!(
            min_x - bracket.0 < 1e-4,
            "min_x - bracket.0 = {}",
            min_x - bracket.0
        );
        assert!(bracket.1 >= min_x);
        assert!(
            bracket.1 - min_x < 1e-4,
            "bracket.1 - min_x = {}",
            bracket.1 - min_x
        );
    }
}
