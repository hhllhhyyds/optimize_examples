use std::collections::VecDeque;
use std::fmt::Debug;
use std::rc::Rc;

use uuid::Uuid;

use super::basic_fn::{BasicFn, FnInput};

#[derive(Clone)]
pub struct Node {
    func: Option<Rc<BasicFn>>,
    pub(super) value: f64,
    parents: Vec<Rc<Node>>,
    id: Uuid,
}

impl Node {
    pub fn new(parents: Vec<Rc<Node>>, value: f64, func: Option<Rc<BasicFn>>) -> Self {
        let parents_empty = parents.is_empty();
        let func_none = func.is_none();
        assert!(!(parents_empty ^ func_none));

        Self {
            parents,
            value,
            func,
            id: Uuid::new_v4(),
        }
    }

    pub fn start(value: f64) -> Self {
        Self::new(vec![], value, None)
    }

    pub fn is_start(&self) -> bool {
        self.parents.is_empty() && self.func.is_none()
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn grad_value(&self) -> Option<Vec<f64>> {
        self.func.as_ref().map(|f| {
            let grads = f.grad_fn()(
                &self
                    .parents
                    .iter()
                    .map(|x| FnInput::from(x.value()))
                    .collect::<Vec<_>>(),
            );
            assert!(grads.len() == self.parents.len());
            grads
        })
    }

    pub fn auto_grad(node: Rc<Node>) -> Vec<NodeGradPair> {
        fn insert_pair(queue: &mut VecDeque<NodeGradPair>, pair: NodeGradPair) {
            for old_pair in queue.iter_mut() {
                if old_pair.node().id() == pair.node().id() {
                    old_pair.grad += pair.grad;
                    return;
                }
            }
            queue.push_back(pair);
        }

        fn process_one(queue: &mut VecDeque<NodeGradPair>) {
            let processed_pair = queue.pop_front().unwrap();
            let processed_node = processed_pair.node().clone();

            if processed_node.is_start() {
                queue.push_back(processed_pair);
                return;
            }

            processed_node
                .parents
                .iter()
                .zip(processed_node.grad_value().unwrap())
                .map(|(node, grad)| NodeGradPair::new(node.clone(), grad * processed_pair.grad))
                .for_each(|pair| insert_pair(queue, pair));
        }

        let mut grads = VecDeque::new();
        grads.push_back(NodeGradPair::new(node, 1.0));

        while grads.iter().any(|pair| !pair.node().is_start()) {
            for _ in 0..grads.len() {
                process_one(&mut grads);
            }
        }

        let mut grads = grads.into_iter().collect::<Vec<_>>();
        grads.sort_by_key(|a| a.node().id());

        grads
    }
}

#[derive(Clone, Debug)]
pub struct NodeGradPair {
    node: Rc<Node>,
    grad: f64,
}

impl NodeGradPair {
    pub fn new(node: Rc<Node>, grad: f64) -> Self {
        Self { node, grad }
    }

    pub fn node(&self) -> &Node {
        &self.node
    }

    pub fn grad(&self) -> f64 {
        self.grad
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ComputationGraph Node: {{ BasicFn: ")?;
        if let Some(func) = &self.func {
            func.fmt(f)?;
        } else {
            write!(f, "none")?;
        }
        write!(f, ", Value: {:.3}", self.value)?;
        writeln!(f, ", Parents: [")?;
        for p in &self.parents {
            p.fmt(f)?;
            write!(f, ", ")?;
        }
        write!(f, "]}}")?;

        Ok(())
    }
}

impl std::ops::Add for Node {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        BasicFn::sum().to_gen_node_fn()(&[Rc::new(self).into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Add<f64> for Node {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        BasicFn::sum().to_gen_node_fn()(&[Rc::new(self).into(), rhs.into()])
    }
}

impl std::ops::Add<Node> for f64 {
    type Output = Node;
    fn add(self, rhs: Node) -> Self::Output {
        BasicFn::sum().to_gen_node_fn()(&[self.into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Sub for Node {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        BasicFn::sub().to_gen_node_fn()(&[Rc::new(self).into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Sub<f64> for Node {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self::Output {
        BasicFn::sub().to_gen_node_fn()(&[Rc::new(self).into(), rhs.into()])
    }
}

impl std::ops::Sub<Node> for f64 {
    type Output = Node;
    fn sub(self, rhs: Node) -> Self::Output {
        BasicFn::sub().to_gen_node_fn()(&[self.into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Mul for Node {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        BasicFn::product().to_gen_node_fn()(&[Rc::new(self).into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Mul<f64> for Node {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        BasicFn::product().to_gen_node_fn()(&[Rc::new(self).into(), rhs.into()])
    }
}

impl std::ops::Mul<Node> for f64 {
    type Output = Node;
    fn mul(self, rhs: Node) -> Self::Output {
        BasicFn::product().to_gen_node_fn()(&[self.into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Div for Node {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        BasicFn::div().to_gen_node_fn()(&[Rc::new(self).into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Div<f64> for Node {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        BasicFn::div().to_gen_node_fn()(&[Rc::new(self).into(), rhs.into()])
    }
}

impl std::ops::Div<Node> for f64 {
    type Output = Node;
    fn div(self, rhs: Node) -> Self::Output {
        BasicFn::div().to_gen_node_fn()(&[self.into(), Rc::new(rhs).into()])
    }
}

impl std::ops::Neg for Node {
    type Output = Self;
    fn neg(self) -> Self::Output {
        BasicFn::neg().to_gen_node_fn()(&[Rc::new(self).into()])
    }
}
