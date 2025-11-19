// frontend/src/utils/animations.ts
import { Variants } from 'framer-motion';

// Variantes de animación reutilizables para componentes

export const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { duration: 0.3 }
  }
};

export const slideInFromTop: Variants = {
  hidden: { opacity: 0, y: -20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.3, ease: 'easeOut' }
  }
};

export const slideInFromBottom: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.3, ease: 'easeOut' }
  }
};

export const slideInFromLeft: Variants = {
  hidden: { opacity: 0, x: -20 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.3, ease: 'easeOut' }
  }
};

export const slideInFromRight: Variants = {
  hidden: { opacity: 0, x: 20 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.3, ease: 'easeOut' }
  }
};

export const scaleIn: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { 
    opacity: 1, 
    scale: 1,
    transition: { duration: 0.2, ease: 'easeOut' }
  }
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    }
  }
};

// Props de animación común para componentes
export const commonAnimationProps = {
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
};

// Variantes de hover para Framer Motion
export const hoverScale: Variants = {
  rest: { scale: 1 },
  hover: { 
    scale: 1.05,
    transition: { duration: 0.2, ease: 'easeOut' }
  },
  tap: { scale: 0.95 }
};

export const hoverLift: Variants = {
  rest: { y: 0, boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)' },
  hover: { 
    y: -4,
    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
    transition: { duration: 0.2, ease: 'easeOut' }
  }
};

export const hoverSlide: Variants = {
  rest: { x: 0 },
  hover: { 
    x: 4,
    transition: { duration: 0.2, ease: 'easeOut' }
  }
};

export const hoverGlow: Variants = {
  rest: { 
    boxShadow: '0 0 0px rgba(26, 128, 255, 0)',
    borderColor: 'transparent'
  },
  hover: { 
    boxShadow: '0 0 20px rgba(26, 128, 255, 0.5)',
    borderColor: 'rgba(26, 128, 255, 0.6)',
    transition: { duration: 0.3, ease: 'easeOut' }
  }
};

export const hoverRotate: Variants = {
  rest: { rotate: 0 },
  hover: { 
    rotate: 5,
    transition: { duration: 0.2, ease: 'easeOut' }
  }
};

export const pulse: Variants = {
  rest: { scale: 1, opacity: 1 },
  pulse: {
    scale: [1, 1.05, 1],
    opacity: [1, 0.8, 1],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut'
    }
  }
};

