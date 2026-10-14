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
    ? Math.min(height * .65, Math.max(height * .18, props.subtitleFontSize * 2 + height * .08))
    : height * .07;
  const padding = width * .06;
  const available = height - bottom - height * .08;
  const titleHeight = Math.min(available * .24, height * .16);
  const bodyHeight = available - titleHeight - height * .05;
  const rows = portrait ? props.items.length : Math.ceil(props.items.length / 2);
  const gap = Math.min(height * .035, bodyHeight * .04);
  const cardPadding = Math.min(32, bodyHeight * .035);
  const comparisonRows = Math.max(1, ...(['left', 'right'] as const).map((column) =>
    props.items.filter((item) => item.column === column).length));
  const panelGap = Math.min(22, bodyHeight * .02);
  const panelHeight = portrait ? (bodyHeight - gap) / 2 : bodyHeight;
  const comparisonFontSize = (panelHeight - cardPadding * 2 - comparisonRows * panelGap)
    / ((comparisonRows + .8) * 1.3);
  const fontSize = Math.min(portrait ? 74 : 68,
    props.template === 'compare' ? comparisonFontSize
      : bodyHeight / (props.template === 'flow' ? Math.max(1, rows) * 3.3 : 3.3));
  const opacity = interpolate(frame, [0, 7, durationInFrames - 8, durationInFrames - 1],
    [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const entrance = (at: number) => spring({frame: frame - at, fps, config: {damping: 22, stiffness: 130}});
  const itemStyle = (item: Item): React.CSSProperties => ({
    opacity: interpolate(frame - item.atFrame, [0, 7], [0, 1],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    transform: `translateY(${(1 - entrance(item.atFrame)) * 20}px)`,
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
    <div style={{position: 'absolute', top: height * .08, left: padding, right: padding,
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
            ...line(props.title, width - padding * 2, Math.min(portrait ? 84 : 88, titleHeight * .65)), fontWeight: 700}}>{props.title}</div>
          {props.template === 'flow'
            ? <div style={{display: 'grid', gap, flex: 1, minHeight: 0,
              gridTemplateColumns: portrait ? '1fr' : '1fr 1fr',
              gridAutoRows: '1fr'}}>
              {props.items.map((item, index) => <div key={index} style={{...itemStyle(item),
                border: `2px solid ${props.accent}70`, borderRadius: 22, backgroundColor: '#152233',
                position: 'relative', padding: cardPadding, display: 'flex', flexDirection: 'column', justifyContent: 'center',
                gridColumn: !portrait && props.items.length === 3 && index === 2 ? '1 / -1'
                  : !portrait && props.items.length === 4 && index >= 2 ? (index === 2 ? 2 : 1) : undefined,
                gridRow: !portrait && props.items.length === 4 && index >= 2 ? 2 : undefined}}>
                <div style={{color: props.accent, fontSize: fontSize * .52, marginBottom: Math.min(12, bodyHeight * .02)}}>{String(index + 1).padStart(2, '0')}</div>
                <div style={line(item.text, (width - padding * 2) / (portrait || (props.items.length === 3 && index === 2) ? 1 : 2) - 90,
                  fontSize)}>{item.text}</div>
                {index < props.items.length - 1 && <div style={{position: 'absolute', color: props.accent,
                  fontSize: Math.min(fontSize * .5, gap * .8), zIndex: 1,
                  opacity: interpolate(frame - props.items[index + 1].atFrame, [0, 7], [0, 1],
                    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
                  ...(portrait || index % 2 === 1
                    ? {left: '50%', bottom: -gap / 2, transform: 'translate(-50%, 50%)'}
                    : index === 2 ? {left: -gap / 2, top: '50%', transform: 'translate(-50%, -50%)'}
                      : {right: -gap / 2, top: '50%', transform: 'translate(50%, -50%)'})}}>
                  {portrait || index % 2 === 1 ? '↓' : index === 2 ? '←' : '→'}
                </div>}
              </div>)}
            </div>
            : <div style={{display: 'grid', flex: 1, minHeight: 0, gap,
              gridTemplateColumns: portrait ? '1fr' : '1fr 1fr'}}>
              {(['left', 'right'] as const).map((column) => {
                const heading = column === 'left' ? props.leftTitle : props.rightTitle;
                const room = (width - padding * 2) / (portrait ? 1 : 2) - 80;
                return <div key={column} style={{backgroundColor: '#152233', borderRadius: 22,
                  padding: cardPadding, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: panelGap}}>
                  <div style={{...line(heading, room, fontSize * .8), color: props.accent}}>{heading}</div>
                  {props.items.filter((item) => item.column === column).map((item, index) =>
                    <div key={index} style={{...itemStyle(item), ...line(item.text, room, fontSize)}}>{item.text}</div>)}
                </div>;
              })}
            </div>}
        </>}
    </div>
    <div style={{position: 'absolute', left: padding, right: padding,
      bottom: bottom + height * .02, height: 3, backgroundColor: '#263447'}}>
      <div style={{height: '100%', width: `${100 * frame / Math.max(1, durationInFrames - 1)}%`, backgroundColor: props.accent}} />
    </div>
  </AbsoluteFill>;
};

const Root: React.FC = () => <Composition id="Explanation" component={Explanation}
  width={1920} height={1080} fps={30} durationInFrames={240} defaultProps={defaults}
  calculateMetadata={({props}) => ({width: props.width, height: props.height,
    fps: 30, durationInFrames: props.frames})} />;

registerRoot(Root);
