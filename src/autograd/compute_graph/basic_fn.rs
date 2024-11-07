use std::fmt::Debug;
use std::rc::Rc;

use super::node::Node;

#[derive(Debug, Clone)]
pub enum FnInput {
    Float(f64),
    Node(Rc<Node>),
}

impl From<f64> for FnInput {
    fn from(value: f64) -> Self {
        FnInput::Float(value)
    }
}

impl From<Rc<Node>> for FnInput {
    fn from(value: Rc<Node>) -> Self {
        FnInput::Node(value)
    }
}

impl From<Node> for FnInput {
    fn from(value: Node) -> Self {
        Self::from(Rc::new(value))
    }
}

impl AsRef<f64> for FnInput {
    fn as_ref(&self) -> &f64 {
        match self {
            FnInput::Float(x) => x,
            FnInput::Node(x) => &x.value,
        }
    }
}

type FloatFnMultiToSingle = Rc<dyn Fn(&[FnInput]) -> f64>;
type FloatFnMultiToMulti = Rc<dyn Fn(&[FnInput]) -> Vec<f64>>;
type FloatFnMultiToNode<'a> = Rc<dyn Fn(&[FnInput]) -> Node + 'a>;

#[derive(Clone)]
pub struct BasicFn {
    func: FloatFnMultiToSingle,
    grad: FloatFnMultiToMulti,
    info: Option<String>,
}

impl BasicFn {
    pub fn new(func: FloatFnMultiToSingle, grad: FloatFnMultiToMulti) -> Self {
        Self {
            func,
            grad,
            info: None,
        }
    }

    pub fn with_info(mut self, info: &str) -> Self {
        self.info = Some(info.to_string());
        self
    }

    pub fn value_fn(&self) -> FloatFnMultiToSingle {
        self.func.clone()
    }

    pub fn grad_fn(&self) -> FloatFnMultiToMulti {
        self.grad.clone()
    }

    pub fn to_gen_node_fn(&self) -> FloatFnMultiToNode {
        Rc::new(|inputs: &[FnInput]| -> Node {
            let parents = inputs
                .iter()
                .filter_map(|x| {
                    if let FnInput::Node(node) = x {
                        Some(node.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let value = (*self.func)(inputs);

            Node::new(parents, value, Some(Rc::new(self.clone())))
        })
    }

    pub fn exp() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                inputs[0].as_ref().exp()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                vec![inputs[0].as_ref().exp()]
            }),
        )
        .with_info("exp")
    }

    pub fn ln() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                inputs[0].as_ref().ln()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                vec![1.0 / inputs[0].as_ref()]
            }),
        )
        .with_info("ln")
    }

    pub fn sin() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                inputs[0].as_ref().sin()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                vec![inputs[0].as_ref().cos()]
            }),
        )
        .with_info("sin")
    }

    pub fn cos() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                inputs[0].as_ref().cos()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                vec![-inputs[0].as_ref().sin()]
            }),
        )
        .with_info("sin")
    }

    pub fn neg() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                -inputs[0].as_ref()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 1);
                vec![-1.0]
            }),
        )
        .with_info("neg")
    }

    pub fn sum() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() > 1);
                inputs.iter().map(|input| input.as_ref()).sum()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() > 1);
                vec![1.0; inputs.len()]
            }),
        )
        .with_info("sum")
    }

    pub fn sub() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 2);
                inputs[0].as_ref() - inputs[1].as_ref()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 2);
                vec![1.0, -1.0]
            }),
        )
        .with_info("sub")
    }

    pub fn product() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() > 1);
                inputs.iter().map(|input| input.as_ref()).product()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() > 1);
                let mut grad = vec![];
                #[allow(clippy::needless_range_loop)]
                for i in 0..inputs.len() {
                    let mut p = 1.0;
                    for j in 0..inputs.len() {
                        if j != i {
                            p *= inputs[j].as_ref();
                        }
                    }
                    grad.push(p);
                }
                grad
            }),
        )
        .with_info("product")
    }

    pub fn div() -> Self {
        BasicFn::new(
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 2);
                inputs[0].as_ref() / inputs[1].as_ref()
            }),
            Rc::new(|inputs| {
                debug_assert!(inputs.len() == 2);
                vec![
                    1.0 / inputs[1].as_ref(),
                    -inputs[0].as_ref() / inputs[1].as_ref().powi(2),
                ]
            }),
        )
        .with_info("div")
    }
}

impl Debug for BasicFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(s) = &self.info {
            write!(f, "{s}")?;
        } else {
            write!(f, "ComputeGraph BasicFn Instance")?;
        }
        Ok(())
    }
}
