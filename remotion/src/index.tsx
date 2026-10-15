import React, {useEffect, useState} from 'react';
import {
  AbsoluteFill, Composition, cancelRender, continueRender, delayRender,
  interpolate, registerRoot, spring, useCurrentFrame, useVideoConfig,
} from 'remotion';

type Item = {text: string; atFrame: number; column: 'left' | 'right'};
type Props = {
  template: 'chapter' | 'flow' | 'compare';
  title: string;
  leftTitle: string;
  rightTitle: string;
  items: Item[];
  width: number;
  height: number;
  frames: number;
  accent: string;
  subtitleFontSize: number;
  showSubtitles: boolean;
  fontData: string;
};

const defaults: Props = {
  template: 'flow', title: '关系与责任', leftTitle: '观点一', rightTitle: '观点二',
  items: [{text: '进入生活', atFrame: 10, column: 'left'},
    {text: '建立关系', atFrame: 60, column: 'left'},
    {text: '承担责任', atFrame: 120, column: 'left'}],
  width: 1920, height: 1080, frames: 240, accent: '#66D5B8',
  subtitleFontSize: 72, showSubtitles: true, fontData: '',
};

const textSize = (text: string, width: number, preferred: number) =>
  Math.min(preferred, width / Math.max(1, Array.from(text).length));

const Explanation: React.FC<Props> = (props) => {
  const frame = useCurrentFrame();
  const {width, height, fps, durationInFrames} = useVideoConfig();
  const portrait = height > width;
  const [fontHandle] = useState(() => delayRender('载入本地动画字体'));
  useEffect(() => {
    let active = true;
    const load = async () => {
      if (props.fontData) {
        const font = new FontFace('NarratoMotion', `url(${props.fontData})`);
        await font.load();
        document.fonts.add(font);
      }
      if (active) continueRender(fontHandle);
    };
    load().catch((error: unknown) => cancelRender(error));
    return () => {active = false;};
  }, [fontHandle, props.fontData]);

  const bottom = props.showSubtitles
    ? Math.max(height * .18, props.subtitleFontSize * 2 + height * .08)
    : height * .07;
  const padding = width * .06;
  const contentWidth = width - padding * 2;
  const contentTop = height * .08;
  const progressHeight = Math.max(3, height * .003);
  const progressTop = height - bottom - progressHeight;
  // Cards end before a separate progress row; subtitle space is never shared.
  const available = progressTop - height * .025 - contentTop;
  if (!Number.isFinite(available) || available < height * .22) {
    throw new Error('字幕字号过大，解释动画没有足够空间；请减小字幕字号或关闭解释动画。');
  }
  const titleHeight = Math.min(available * .22, height * .14);
  const titleGap = Math.min(height * .035, available * .06);
  const bodyHeight = available - titleHeight - titleGap;
  const rows = Math.max(1, portrait ? props.items.length : Math.ceil(props.items.length / 2));
  const gap = Math.min(height * .035, bodyHeight * .04);
  const cardHeight = (bodyHeight - (rows - 1) * gap) / rows;
  const cardPadding = Math.min(28, cardHeight * .12);
  const numberGap = Math.min(12, cardHeight * .04);
  const boxes = props.items.map((_, index) => {
    const fullWidth = portrait || (props.items.length === 3 && index === 2);
    const row = portrait ? index : Math.floor(index / 2);
    const column = !portrait && props.items.length === 4 && index >= 2 ? 3 - index : index % 2;
    const cardWidth = fullWidth ? contentWidth : (contentWidth - gap) / 2;
    return {left: fullWidth ? 0 : column * (cardWidth + gap), top: row * (cardHeight + gap),
      width: cardWidth, height: cardHeight};
  });
  const comparisonRows = Math.max(1, ...(['left', 'right'] as const).map((column) =>
    props.items.filter((item) => item.column === column).length));
  const panelGap = Math.min(22, bodyHeight * .02);
  const panelHeight = portrait ? (bodyHeight - gap) / 2 : bodyHeight;
  const comparisonPadding = Math.min(28, panelHeight * .08);
  const comparisonFontSize = (panelHeight - comparisonPadding * 2 - comparisonRows * panelGap)
    / ((comparisonRows + .8) * 1.3);
  const fontSize = Math.min(portrait ? 74 : 68,
    props.template === 'compare' ? comparisonFontSize
      : props.template === 'flow' ? (cardHeight - cardPadding * 2 - numberGap - 4 - Math.min(12, gap * .2) * 2) / 2
        : available / 3.3);
  const opacity = interpolate(frame, [0, 7, durationInFrames - 8, durationInFrames - 1],
    [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const entrance = (at: number) => spring({frame: frame - at, fps, config: {damping: 22, stiffness: 130}});
  const itemStyle = (item: Item): React.CSSProperties => ({
    opacity: interpolate(frame - item.atFrame, [0, 7], [0, 1],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    transform: `translateY(${(1 - Math.min(1, entrance(item.atFrame))) * Math.min(12, gap * .2)}px)`,
  });
  const line = (text: string, room: number, preferred: number, lines = 1): React.CSSProperties => ({
    fontSize: textSize(text, room * lines, preferred), whiteSpace: lines === 1 ? 'nowrap' : 'normal',
    wordBreak: lines === 1 ? 'normal' : 'break-all', lineHeight: 1.3,
  });
  const chapterUnit = available / (6 + Math.max(1, props.items.length) * 3.4);

  return <AbsoluteFill style={{backgroundColor: '#0B1420', color: '#F6F8FA',
    fontFamily: 'NarratoMotion, "Microsoft YaHei", sans-serif'}}>
    <div style={{position: 'absolute', top: height * .04, left: padding,
      width: 90, height: 6, backgroundColor: props.accent}} />
    <div style={{position: 'absolute', top: contentTop, left: padding, right: padding,
      height: available, opacity, display: 'flex', flexDirection: 'column'}}>
      {props.template === 'chapter'
        ? <div style={{margin: 'auto 0'}}>
          <div style={{fontSize: Math.min(fontSize * .55, chapterUnit * .5), color: props.accent,
            marginBottom: Math.min(30, chapterUnit * .3)}}>论述章节</div>
          <div style={{...line(props.title, width - padding * 2,
            Math.min(portrait ? 100 : 120, chapterUnit * 1.6), 2), fontWeight: 700}}>{props.title}</div>
          <div style={{height: 5, width: (width - padding * 2) * entrance(8),
            backgroundColor: props.accent, marginTop: Math.min(40, chapterUnit * .4), maxWidth: 340}} />
          {props.items.map((item, index) => <div key={index} style={{...itemStyle(item),
            ...line(item.text, width - padding * 2, Math.min(fontSize, chapterUnit), 2),
            color: '#ADB8C8', marginTop: Math.min(32, chapterUnit * .3)}}>{item.text}</div>)}
        </div>
        : <>
          <div style={{height: titleHeight, flexShrink: 0,
            marginBottom: titleGap,
            ...line(props.title, contentWidth, Math.min(portrait ? 84 : 88, titleHeight * .65)), fontWeight: 700}}>{props.title}</div>
          {props.template === 'flow'
            ? <div style={{position: 'relative', height: bodyHeight, flexShrink: 0}}>
              {props.items.map((item, index) => <div key={index} style={{...boxes[index],
                opacity: itemStyle(item).opacity, boxSizing: 'border-box',
                border: `2px solid ${props.accent}70`, borderRadius: 22, backgroundColor: '#152233',
                position: 'absolute', padding: cardPadding, display: 'flex', justifyContent: 'center', flexDirection: 'column'}}>
                <div style={{transform: itemStyle(item).transform}}>
                  <div style={{color: props.accent, fontSize: fontSize * .52, lineHeight: 1.3, marginBottom: numberGap}}>{String(index + 1).padStart(2, '0')}</div>
                  <div style={line(item.text, boxes[index].width - cardPadding * 2 - 4, fontSize)}>{item.text}</div>
                </div>
              </div>)}
              <svg width={contentWidth} height={bodyHeight} style={{position: 'absolute', inset: 0, pointerEvents: 'none'}}>
                {boxes.slice(0, -1).map((box, index) => {
                  const next = boxes[index + 1];
                  const vertical = next.top > box.top;
                  const direction = vertical || next.left > box.left ? 1 : -1;
                  const x = box.left + box.width / 2;
                  const y = box.top + box.height / 2;
                  const x1 = vertical ? x : direction > 0 ? box.left + box.width + gap * .22 : box.left - gap * .22;
                  const x2 = vertical ? x : direction > 0 ? next.left - gap * .22 : next.left + next.width + gap * .22;
                  const y1 = vertical ? box.top + box.height + gap * .22 : y;
                  const y2 = vertical ? next.top - gap * .22 : y;
                  const head = Math.min(8, gap * .18);
                  const tip = vertical ? `${x2-head},${y2-head} ${x2},${y2} ${x2+head},${y2-head}`
                    : `${x2-direction*head},${y2-head} ${x2},${y2} ${x2-direction*head},${y2+head}`;
                  return <g key={index} stroke={props.accent} strokeWidth={Math.min(2, gap * .08)} fill="none"
                    opacity={itemStyle(props.items[index + 1]).opacity}>
                    <line x1={x1} y1={y1} x2={x2} y2={y2} />
                    <polyline points={tip} />
                  </g>;
                })}
              </svg>
            </div>
            : <div style={{display: 'grid', height: bodyHeight, flexShrink: 0, gap,
              gridTemplateColumns: portrait ? '1fr' : '1fr 1fr',
              gridTemplateRows: portrait ? 'repeat(2, minmax(0, 1fr))' : 'minmax(0, 1fr)'}}>
              {(['left', 'right'] as const).map((column) => {
                const heading = column === 'left' ? props.leftTitle : props.rightTitle;
                const room = (portrait ? contentWidth : (contentWidth - gap) / 2) - comparisonPadding * 2;
                return <div key={column} style={{backgroundColor: '#152233', borderRadius: 22,
                  boxSizing: 'border-box', padding: comparisonPadding, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: panelGap}}>
                  <div style={{...line(heading, room, fontSize * .8), color: props.accent}}>{heading}</div>
                  {props.items.filter((item) => item.column === column).map((item, index) =>
                    <div key={index} style={{...itemStyle(item), ...line(item.text, room, fontSize)}}>{item.text}</div>)}
                </div>;
              })}
            </div>}
        </>}
    </div>
    <div style={{position: 'absolute', left: padding, right: padding,
      top: progressTop, height: progressHeight, backgroundColor: '#263447'}}>
      <div style={{height: '100%', width: `${100 * frame / Math.max(1, durationInFrames - 1)}%`, backgroundColor: props.accent}} />
    </div>
  </AbsoluteFill>;
};

const Root: React.FC = () => <Composition id="Explanation" component={Explanation}
  width={1920} height={1080} fps={30} durationInFrames={240} defaultProps={defaults}
  calculateMetadata={({props}) => ({width: props.width, height: props.height,
    fps: 30, durationInFrames: props.frames})} />;

registerRoot(Root);
