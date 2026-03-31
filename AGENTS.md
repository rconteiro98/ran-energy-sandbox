## Project
ucl-ran-energy-sandbox

## Goal
Build a modular, MVP for RAN energy optimization around the UCL Bloomsbury campus area.

## Development rules
- Build one module at a time.
- Each module must run independently before integrating with the next one.
- Keep the code simple, readable, and easy to explain in an interview.
- Prefer deterministic and testable implementations.
- Do not overengineer abstractions.
- Add docstrings and concise comments.
- Preserve clean interfaces between modules.
- Use pandas for tabular data work.
- Keep file and function names explicit.

## Module order
1. geo
2. sim
3. rules
4. ml
5. compare
6. llm
7. interface

## Current focus
Implement only the geo module.
Do not create ML, API, dashboard, or benchmark logic yet unless explicitly requested.