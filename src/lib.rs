use pyo3::prelude::*;

type Number = i64;
type Pos = u16;
const MASK: usize = NUM_HASH - 1;
const NUM_HASH: usize = 1 << 9; // 9 bit or 512

// x from 0 to 64 inclusive
const OFFSET_8: Number = 65;
const OFFSET_1: Number = 65 + 8;
const LENGTH: Number = 65 + 16;

#[pyfunction]
fn compress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    if xs.len() < 3 {
        return Ok(xs);
    }

    let mut result = Vec::with_capacity(xs.len());
    let mut chain = HashChain::new(&xs[..], 256);
    let mut pos = 0;
    let mut iter = xs.iter().skip(2);
    while let Some(x) = iter.next() {
        if let Some(match_pos) = chain.insert(*x) {
            let length_pos = chain.get_max_length(match_pos, pos as Pos);
            match length_pos.0 {
                0..=3 => {
                    result.push(xs[pos]);
                    pos += 1;
                }
                length => {
                    let offset = (pos - length_pos.1) as Number;
                    result.push(offset / 8 + OFFSET_8);
                    result.push(offset % 8 + OFFSET_1);
                    result.push(length as Number + LENGTH);
                    for _ in 0..length {
                        iter.next().map(|x| chain.insert(*x));
                        pos += 1;
                    }
                }
            }
        } else {
            result.push(xs[pos]);
            pos += 1;
        }
    }
    for x in &xs[pos..] {
        result.push(*x);
    }
    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    Ok(())
}

struct HashChain<'a> {
    h: usize,
    head: Vec<Pos>,
    tail: Vec<Pos>,
    cur_pos: Pos,
    max_offset: usize,
    xs: &'a [Number],
}

impl HashChain<'_> {
    fn new(xs: &[Number], max_offset: usize) -> HashChain {
        let h = (((xs[0] as usize) << 3) ^ (xs[1] as usize)) & MASK;
        HashChain {
            h,
            head: vec![0 as Pos; NUM_HASH],
            tail: vec![0 as Pos; xs.len()],
            cur_pos: 0,
            max_offset,
            xs,
        }
    }

    // update hash
    // update head & tail
    // returns pos pointed by head
    fn insert(&mut self, x: Number) -> Option<Pos> {
        self.update_hash(x);
        let prev_pos = self.head[self.h];
        self.head[self.h] = self.cur_pos;
        self.tail[self.cur_pos as usize] = prev_pos;
        self.cur_pos += 1;
        if prev_pos == 0 {
            None
        } else {
            Some(prev_pos)
        }
    }

    fn update_hash(&mut self, x: Number) {
        self.h = ((self.h << 3) ^ (x as usize)) & MASK;
    }

    fn next_pos(&self, pos: Pos) -> Option<Pos> {
        match &self.tail[pos as usize] {
            0 => None,
            pos => Some(*pos),
        }
    }

    // returns (max length, pos)
    fn get_max_length(&self, mut pos: Pos, target_pos: Pos) -> (usize, usize) {
        let mut result = (0, 0);
        let target = &self.xs[target_pos as usize..];
        loop {
            if target_pos as usize - pos as usize > self.max_offset {
                break;
            }
            let length = self.xs[pos as usize..]
                .iter()
                .zip(target.iter())
                .map_while(|(x, y)| match x.eq(y) {
                    true => Some(true),
                    false => None,
                })
                .count();
            result = if length >= result.0 {
                (length, pos as usize)
            } else {
                result
            };

            pos = if let Some(match_pos) = self.next_pos(pos) {
                match_pos
            } else {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod test {
    use super::compress;

    #[test]
    fn test() {
        println!("{:?}", compress(vec![0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]));
    }
}
