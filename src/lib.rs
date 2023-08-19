use pyo3::{exceptions::PyTypeError, prelude::*};

type Number = i64;
type Pos = u16;
const MASK: usize = NUM_HASH - 1;
const NUM_HASH: usize = 1 << 9; // 9 bit or 512

// x from 0 to 64 inclusive, i.e., 65 tokens
const OFFSET_8: Number = 65;
const OFFSET_1: Number = OFFSET_8 + 8; // 73
const LENGTH: Number = OFFSET_1 + 8; // 81
const NUM_TOKENS: Number = LENGTH + 16; // 98

#[pyfunction]
fn compress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    if xs.len() < 3 {
        return Ok(xs);
    }

    let mut result = Vec::with_capacity(xs.len());
    let mut chain = HashChain::new(&xs[..], 64);
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

#[pyfunction]
fn decompress(xs: Vec<Number>) -> PyResult<Vec<Number>> {
    let mut result = Vec::new();
    let mut iter = xs.iter();
    while let Some(x) = iter.next() {
        let x = *x;
        if x < OFFSET_8 {
            // literal
            result.push(x);
        } else if x < OFFSET_1 {
            let mut offset = (x - OFFSET_8) << 3;
            match (iter.next(), iter.next()) {
                (Some(y), Some(z))
                    if *y >= OFFSET_1 && *y < LENGTH && *z >= LENGTH && *z < NUM_TOKENS =>
                {
                    offset += *y - OFFSET_1;
                    let length = *z - LENGTH;
                    for _ in 0..length {
                        result.push(xs[result.len() - offset as usize]);
                    }
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Decompressor Error: Invalid error {}...",
                        x
                    )))
                }
            }
        }
    }

    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn compressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
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
    type Result<R> = std::result::Result<R, Box<dyn std::error::Error>>;
    use crate::OFFSET_8;

    use super::compress;
    use super::decompress;
    use rand::prelude::*;

    #[test]
    fn test() -> Result<()> {
        let x = vec![0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6];
        let compressed = compress(x.clone())?;
        let decompressed = decompress(compressed)?;
        assert_eq!(x, decompressed);
        Ok(())
    }

    #[test]
    fn test_random() -> Result<()> {
        pyo3::prepare_freethreaded_python();
        let mut rng = thread_rng();
        let distr = rand::distributions::Uniform::new(0, OFFSET_8);
        let length_distr = rand::distributions::Uniform::new(0, 256);
        for _ in 0..1000 {
            let n = rng.sample(length_distr);
            let mut xs = Vec::new();
            for _ in 0..n {
                xs.push(rng.sample(distr));
            }
            assert_eq!(decompress(compress(xs.clone())?)?, xs);
        }

        Ok(())
    }

    #[test]
    fn test_case() -> Result<()> {
        let xs = vec![
            50, 27, 38, 62, 0, 37, 11, 18, 23, 1, 24, 18, 47, 41, 27, 13, 35, 25, 14, 58, 24, 28,
            50, 26, 25, 7, 58, 57, 24, 9, 39, 54, 33, 39, 15, 32, 31, 58, 64, 7, 40, 27, 17, 28,
            61, 25, 48, 26, 37, 47, 36, 50, 63, 33, 49, 46, 19, 13, 33, 16, 13, 40, 23, 25, 22, 37,
            2, 56, 1, 49, 34, 36, 17, 3, 17, 7, 58, 42, 56, 54, 31, 17, 38, 40, 18, 4, 48, 10, 20,
            5, 13, 32, 17, 21, 26, 50, 14, 23, 1, 2, 5, 50, 38, 42, 25, 17, 51, 52, 8, 21, 45, 45,
            48, 34, 4, 40, 6, 37, 18, 12, 26, 16, 39, 55, 2, 34, 33, 64, 34, 16, 13, 40, 23, 59,
        ];
        let c = compress(xs.clone())?;
        let d = decompress(c.clone())?;
        println!{"{:?}", c}
        assert_eq!(xs, d);
        Ok(())
    }
}
