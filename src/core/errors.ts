export type ErrorCode =
  | 'VALIDATION'
  | 'NOT_FOUND'
  | 'CONFLICT'
  | 'NO_PATH'
  | 'BAD_TAG'
  | 'EMPTY_CORPUS'
  | 'INSUFFICIENT_DATA';

export class AppError extends Error {
  readonly code: ErrorCode;
  readonly details?: unknown;

  constructor(code: ErrorCode, message: string, details?: unknown) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.details = details;
  }
}

export function httpStatusFor(code: ErrorCode): number {
  switch (code) {
    case 'VALIDATION':
    case 'NO_PATH':
    case 'BAD_TAG':
      return 400;
    case 'NOT_FOUND':
      return 404;
    case 'CONFLICT':
      return 409;
    case 'EMPTY_CORPUS':
    case 'INSUFFICIENT_DATA':
      return 422;
    default:
      return 500;
  }
}
